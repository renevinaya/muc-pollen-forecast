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

LAG_FEATURES = [
    "pollen_lag_1",  # yesterday
    "pollen_lag_2",
    "pollen_lag_3",
    "pollen_rolling_3",  # 3-day rolling mean
    "pollen_rolling_7",  # 7-day rolling mean
]

FORECAST_DAYS = 5

FEATURE_COLS = WEATHER_FEATURES + CALENDAR_FEATURES + LAG_FEATURES


class PollenLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


def value_to_level(value: float) -> PollenLevel:
    """Convert a numeric pollen value to a categorical level."""
    if value <= 0:
        return PollenLevel.NONE
    if value <= 10:
        return PollenLevel.LOW
    if value <= 50:
        return PollenLevel.MODERATE
    if value <= 200:
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
class DayForecast:
    date: str
    species: list[SpeciesForecast] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "species": [s.to_dict() for s in self.species],
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
