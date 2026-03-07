"""Data models for the pollen forecast system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# Munich pollen station
LOCATION = "DEMUNC"
LAT = 48.1351
LON = 11.5820

# All pollen species tracked by the LGL Bayern station
ALL_SPECIES = [
    "Alnus",       # Alder
    "Ambrosia",    # Ragweed
    "Artemisia",   # Mugwort
    "Betula",      # Birch
    "Corylus",     # Hazel
    "Fraxinus",    # Ash
    "Poaceae",     # Grasses
    "Populus",     # Poplar
    "Quercus",     # Oak
    "Salix",       # Willow
    "Urtica",      # Nettle
]

# Feature columns used by the model (order matters for training/prediction)
WEATHER_FEATURES = [
    "temperature_max",
    "temperature_min",
    "temperature_mean",
    "precipitation_sum",
    "wind_speed_max",
    "wind_direction",
    "humidity_mean",
    "sunshine_duration",
    "shortwave_radiation_sum",
]

CALENDAR_FEATURES = [
    "day_of_year",
    "day_of_year_sin",
    "day_of_year_cos",
    "month",
]

WINDOW_FEATURES = [
    "hour_of_day",
    "hour_sin",
    "hour_cos",
]

SEASON_FEATURE = [
    "season_active",  # 1.0 if species is in its pollen season, 0.0 otherwise
]

# Weather-derived features (computed from history at training/prediction time)
WEATHER_DERIVED_FEATURES = [
    "gdd",                  # Growing Degree Days (cumulative from Jan 1)
    "temp_rolling_3d",      # 3-day rolling mean temperature
    "temp_rolling_7d",      # 7-day rolling mean temperature
    "sunshine_rolling_3d",  # 3-day rolling mean sunshine duration
    "sunshine_rolling_7d",  # 7-day rolling mean sunshine duration
    "rain_rolling_3d",      # 3-day cumulative precipitation
    "rain_rolling_7d",      # 7-day cumulative precipitation
    "temp_delta_1d",        # Day-over-day temperature change
    "temp_delta_3d",        # 3-day temperature change
    "temp_x_sunshine",      # Interaction: warm & sunny = peak dispersal
    "dry_warm",             # Interaction: warm + low humidity
    # --- Burst potential features (#2) ---
    "gdd_above_threshold",  # max(0, gdd - species GDD threshold)
    "cold_to_warm_flip",    # rapid warming from cold → warm while GDD ready
    "consecutive_warm_hrs", # consecutive 3h windows with temp > activation
    # --- Explosion likelihood features (#6) ---
    "dry_streak",           # consecutive windows with precip ≈ 0
    "warm_after_cold",      # recent warming: temp_rolling_3d - temp_rolling_7d
    "wind_x_dry_warm",      # wind × dry_warm interaction (dispersal capacity)
    # --- Upwind transport features ---
    "wind_dir_sin",         # sin(wind_direction) — E/W component
    "wind_dir_cos",         # cos(wind_direction) — N/S component
    "wind_from_south",      # strength of southerly wind (alpine forests → Munich)
    "wind_from_north",      # strength of northerly wind (plains → Munich)
    "transport_south",      # wind_speed × wind_from_south (pollen transport potential)
    "transport_north",      # wind_speed × wind_from_north
]

LAG_FEATURES = [
    "pollen_lag_1",       # previous 3h window (log-transformed)
    "pollen_lag_2",       # 2 windows ago (6h)
    "pollen_lag_3",       # 3 windows ago (9h)
    "pollen_lag_8",       # same time yesterday (24h)
    "pollen_lag_16",      # 48h ago (#3)
    "pollen_lag_56",      # 7 days ago (#3)
    "pollen_rolling_8",   # 24h rolling mean (log-transformed)
    "pollen_rolling_56",  # 7-day rolling mean (log-transformed)
    "pollen_max_8",       # 24h rolling max (captures recent spikes) (#3)
    "pollen_max_56",      # 7-day rolling max (#3)
    "days_since_active",  # windows since pollen was last > 0 (#3)
]

# NDVI / vegetation features (from MODIS satellite data)
NDVI_FEATURES = [
    "ndvi",         # Normalized Difference Vegetation Index (0–1)
    "evi",          # Enhanced Vegetation Index
    "ndvi_delta",   # Daily NDVI change rate (green-up speed)
]

# Phenology features (from DWD multi-year flowering onset data)
PHENOLOGY_FEATURES = [
    "days_since_typical_onset",  # days since mean flowering onset for this species
    "onset_anomaly",             # current year onset vs historical mean (needs runtime)
]

FORECAST_DAYS = 5
WINDOWS_PER_DAY = 8  # 3-hour windows: 00, 03, 06, 09, 12, 15, 18, 21

FEATURE_COLS = (
    WEATHER_FEATURES
    + CALENDAR_FEATURES
    + WINDOW_FEATURES
    + SEASON_FEATURE
    + WEATHER_DERIVED_FEATURES
    + NDVI_FEATURES
    + PHENOLOGY_FEATURES
    + LAG_FEATURES
)

# GDD base temperature (°C) — standard for temperate deciduous phenology
GDD_T_BASE = 5.0

# Species-specific GDD thresholds for pollen burst readiness (#2)
# When cumulative GDD exceeds this, species is primed for explosive release.
SPECIES_GDD_THRESHOLD: dict[str, float] = {
    "Alnus":     30.0,
    "Ambrosia":  800.0,
    "Artemisia": 700.0,
    "Betula":    150.0,
    "Corylus":   20.0,
    "Fraxinus":  150.0,
    "Poaceae":   400.0,
    "Populus":   100.0,
    "Quercus":   250.0,
    "Salix":     100.0,
    "Urtica":    400.0,
}

# Species-specific activation temperatures for warm-window counting (#2)
SPECIES_ACTIVATION_TEMP: dict[str, float] = {
    "Alnus":     5.0,
    "Ambrosia":  18.0,
    "Artemisia": 16.0,
    "Betula":    10.0,
    "Corylus":   5.0,
    "Fraxinus":  10.0,
    "Poaceae":   12.0,
    "Populus":   8.0,
    "Quercus":   12.0,
    "Salix":     8.0,
    "Urtica":    12.0,
}

# Default thresholds for unknown species
_DEFAULT_GDD_THRESHOLD = 200.0
_DEFAULT_ACTIVATION_TEMP = 10.0

# Pollen season windows per species: (start_month, end_month) inclusive.
# Outside this window the model should predict ~0.
SPECIES_SEASON: dict[str, tuple[int, int]] = {
    "Alnus":     (1, 4),    # January – April
    "Ambrosia":  (7, 10),   # July – October
    "Artemisia": (7, 9),    # July – September
    "Betula":    (3, 5),    # March – May
    "Corylus":   (1, 4),    # January – April
    "Fraxinus":  (3, 5),    # March – May
    "Poaceae":   (5, 9),    # May – September
    "Populus":   (3, 5),    # March – May
    "Quercus":   (4, 6),    # April – June
    "Salix":     (3, 5),    # March – May
    "Urtica":    (5, 9),    # May – September
}


def is_season_active(species: str, month: int) -> bool:
    """Check if a species is within its pollen season for a given month."""
    window = SPECIES_SEASON.get(species)
    if window is None:
        return True  # unknown species: assume always active
    start, end = window
    if start <= end:
        return start <= month <= end
    # wraps around year (e.g., Nov–Feb)
    return month >= start or month <= end


class PollenLevel(str, Enum):
    """Categorical pollen concentration level."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Species-specific pollen level thresholds (DWD / ePIN classifications).
# Tuple: (low_max, moderate_max, high_max). Above high_max → VERY_HIGH.
SPECIES_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "Alnus":     (10,  70,  250),
    "Ambrosia":  (5,   20,   80),
    "Artemisia": (5,   15,   50),
    "Betula":    (10,  50,  300),
    "Corylus":   (10,  70,  250),
    "Fraxinus":  (10,  50,  200),
    "Poaceae":   (5,   30,   60),
    "Populus":   (10,  50,  200),
    "Quercus":   (10,  50,  200),
    "Salix":     (5,   20,   50),
    "Urtica":    (10,  50,  200),
}

# Default thresholds when species is unknown
_DEFAULT_THRESHOLDS = (10, 50, 200)


def value_to_level(value: float, species: str | None = None) -> PollenLevel:
    """Convert a numeric pollen value to a categorical level.

    When *species* is given, use DWD species-specific thresholds.
    """
    if value <= 0:
        return PollenLevel.NONE
    low_max, mod_max, high_max = SPECIES_THRESHOLDS.get(
        species, _DEFAULT_THRESHOLDS
    ) if species else _DEFAULT_THRESHOLDS
    if value <= low_max:
        return PollenLevel.LOW
    if value <= mod_max:
        return PollenLevel.MODERATE
    if value <= high_max:
        return PollenLevel.HIGH
    return PollenLevel.VERY_HIGH


@dataclass
class SpeciesForecast:
    """Forecast for a single species in one time window."""
    name: str
    level: str
    value: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "level": self.level,
            "value": round(self.value, 1),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class WindowForecast:
    """Forecast for a 3-hour time window."""
    from_time: str
    to_time: str
    species: list[SpeciesForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "from": self.from_time,
            "to": self.to_time,
            "species": [s.to_dict() for s in self.species],
        }


@dataclass
class DayForecast:
    """Forecast for a single calendar day."""
    date: str
    windows: list[WindowForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "date": self.date,
            "windows": [w.to_dict() for w in self.windows],
        }


@dataclass
class ForecastOutput:
    """Top-level forecast output with metadata and daily forecasts."""
    generated: str
    location: str
    forecast: list[DayForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "generated": self.generated,
            "location": self.location,
            "forecast": [d.to_dict() for d in self.forecast],
        }

    def to_web_dict(self) -> dict[str, Any]:
        """Serialize to a webapp-compatible dict matching the LGL measurement format.

        Restructures from window-centric (date→window→species) to
        species-centric (species→windows) with unix timestamps in seconds,
        matching the format returned by the ePIN LGL Bayern API.
        """
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("Europe/Berlin")
        species_data: dict[str, list[dict[str, Any]]] = {}

        for day in self.forecast:
            base_date = datetime.strptime(day.date, "%Y-%m-%d")
            for window in day.windows:
                from_h, from_m = map(int, window.from_time.split(":"))
                to_h, to_m = map(int, window.to_time.split(":"))

                from_dt = base_date.replace(
                    hour=from_h, minute=from_m, tzinfo=tz
                )
                if to_h == 0 and to_m == 0:
                    to_dt = (base_date + timedelta(days=1)).replace(tzinfo=tz)
                else:
                    to_dt = base_date.replace(
                        hour=to_h, minute=to_m, tzinfo=tz
                    )

                from_unix = int(from_dt.timestamp())
                to_unix = int(to_dt.timestamp())

                for sp in window.species:
                    species_data.setdefault(sp.name, []).append({
                        "from": from_unix,
                        "to": to_unix,
                        "value": round(sp.value, 1),
                    })

        measurements = [
            {"polle": name, "location": self.location, "data": data}
            for name, data in sorted(species_data.items())
        ]

        return {
            "generated": self.generated,
            "location": self.location,
            "measurements": measurements,
        }
