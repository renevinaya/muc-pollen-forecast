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
]

LAG_FEATURES = [
    "pollen_lag_1",       # previous 3h window (log-transformed)
    "pollen_lag_2",       # 2 windows ago (6h)
    "pollen_lag_3",       # 3 windows ago (9h)
    "pollen_lag_8",       # same time yesterday (24h)
    "pollen_rolling_8",   # 24h rolling mean (log-transformed)
    "pollen_rolling_56",  # 7-day rolling mean (log-transformed)
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
    else:  # wraps around year (e.g., Nov–Feb)
        return month >= start or month <= end


class PollenLevel(str, Enum):
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
    name: str
    level: str
    value: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level,
            "value": round(self.value, 1),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class WindowForecast:
    from_time: str
    to_time: str
    species: list[SpeciesForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_time,
            "to": self.to_time,
            "species": [s.to_dict() for s in self.species],
        }


@dataclass
class DayForecast:
    date: str
    windows: list[WindowForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "windows": [w.to_dict() for w in self.windows],
        }


@dataclass
class ForecastOutput:
    generated: str
    location: str
    forecast: list[DayForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "location": self.location,
            "forecast": [d.to_dict() for d in self.forecast],
        }
