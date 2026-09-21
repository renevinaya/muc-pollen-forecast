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
    "Fungus",      # Mould spores — the ePIN samplers' fungal-spore aggregate
]

# Taxa that are not pollen. Kept in ALL_SPECIES because the whole pipeline is
# per taxon and the same model form works, but they have no DWD index to
# blend with and no flowering season to gate on.
SPORE_TAXA = {"Fungus"}

# Feature columns used by the model (order matters for training/prediction)
# Raw weather columns the collector writes and the feature pipeline carries.
# This is NOT the model's input list — several of these exist only so the
# derived features can be computed from them. Pruning a column out of
# WEATHER_FEATURES below must not remove it here: dropping `wind_direction`
# from what gets carried, for instance, makes _add_weather_derived_features
# silently fall back to a constant 0, which turns wind_dir_sin/cos and both
# transport features into constants without any error.
WEATHER_COLUMNS = [
    "temperature_max",
    "temperature_min",
    "temperature_mean",
    "precipitation_sum",
    "wind_speed_max",
    "wind_direction",
    "humidity_mean",
    "sunshine_duration",
    "shortwave_radiation_sum",
    "boundary_layer_height",
    "dew_point_mean",
    "cape_max",
    "direct_radiation_sum",
    "is_day",
    "temp_slope_3h",
    "humidity_slope_3h",
    "temp_variance_3h",
    "soil_temperature_mean",
    "soil_moisture_mean",
]

# The weather columns the model actually reads. Six of WEATHER_COLUMNS are
# deliberately absent — see the pruning note at FEATURE_COLS.
WEATHER_FEATURES = [
    "temperature_mean",
    "precipitation_sum",
    "wind_speed_max",
    "humidity_mean",
    "sunshine_duration",
    # Diurnal weather features
    "boundary_layer_height",  # PBL height — pollen disperses when PBL rises
    "dew_point_mean",         # dew point — morning dew suppresses pollen release
    "is_day",                 # binary day/night — release is almost entirely daytime
    # Finer resolution features (computed from hourly data within 3h windows)
    "temp_slope_3h",          # temperature change within window (warming ramp signal)
    "humidity_slope_3h",      # humidity change within window (rapid drying triggers release)
    "temp_variance_3h",       # temperature variance within window (changing conditions)
    # Soil features — better predictors of herbaceous/grass onset than air temp
    "soil_temperature_mean",  # soil temp — drives root-zone phenology
    "soil_moisture_mean",     # soil moisture — wet soil delays grass flowering
]

CALENDAR_FEATURES = [
    "day_of_year_sin",
    "day_of_year_cos",
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
    # gdd_above_threshold moved to PHENOLOGY_FEATURES (it now reads the
    # species' selected forcing rule, not the base-5 gdd column) and
    # cold_to_warm_flip was dropped: it earned at most 0.05% of gain for any
    # tree species once the forcing was calibrated per species.
    "consecutive_warm_hrs", # consecutive 3h windows with temp > activation
    # --- Explosion likelihood features (#6) ---
    "dry_streak",           # consecutive windows with precip ≈ 0
    "warm_after_cold",      # recent warming: temp_rolling_3d - temp_rolling_7d
    "wind_x_dry_warm",      # wind × dry_warm interaction (dispersal capacity)
    # --- Upwind transport features ---
    # wind_from_south/north and transport_north were dropped: they are
    # rectified halves of wind_dir_cos, so the pair below plus one transport
    # term carries the same information.
    "wind_dir_sin",         # sin(wind_direction) — E/W component
    "wind_dir_cos",         # cos(wind_direction) — N/S component
    "transport_south",      # wind_speed × southerly strength (transport potential)
]

LAG_FEATURES = [
    "pollen_lag_1",       # previous 3h window (log-transformed)
    "pollen_lag_2",       # 2 windows ago (6h)
    "pollen_lag_3",       # 3 windows ago (9h)
    "pollen_lag_8",       # same time yesterday (24h)
    "pollen_lag_16",      # 48h ago (#3)
    "pollen_lag_24",      # same time 3 days ago (72h) — diurnal patterns recur
    "pollen_lag_56",      # 7 days ago (#3)
    "pollen_rolling_8",   # 24h rolling mean (log-transformed)
    "pollen_rolling_56",  # 7-day rolling mean (log-transformed)
    "pollen_max_8",       # 24h rolling max (captures recent spikes) (#3)
    "pollen_max_56",      # 7-day rolling max (#3)
    "pollen_morning_avg", # mean of today's earlier windows (intra-day trend)
    "days_since_active",  # windows since pollen was last > 0 (#3)
]

# NDVI / vegetation features (from MODIS satellite data)
# EVI is dropped: it is a second index over the same 250 m pixel and moves with
# NDVI, so it earned 1.1% of gain for no information NDVI did not already have.
NDVI_FEATURES = [
    "ndvi",         # Normalized Difference Vegetation Index (0–1)
    "ndvi_delta",   # Daily NDVI change rate (green-up speed)
]

# Phenology features (from DWD multi-year flowering onset data)
PHENOLOGY_FEATURES = [
    "days_since_typical_onset",  # days since this year's causal onset estimate
    "onset_anomaly",             # (forcing − threshold) / threshold under the species' rule
    "gdd_above_threshold",       # max(0, forcing − threshold) under the species' rule
]

# Interannual season load (src/season_load.py): how heavy the previous
# seasons were, as log ratios against the species' own history, so 0 is
# "average or unknown". The only features that cross a season boundary.
LOAD_FEATURES = [
    "load_prev_anom",   # last season vs the seasons before it
    "load_2y_anom",     # last two seasons vs the seasons before them
    "load_trend",       # last season vs the one before (alternation)
]

# CAMS features (optional — from the Copernicus European pollen forecast).
# When the CAMS integration is not activated (no ADS key / deps), this column
# is simply 0 everywhere and the model ignores it. See src/cams.py.
# Empty on purpose. `cams_pollen` was in the model's input list for its whole
# life without ever being populated: it is NaN in 97% of history rows and 0.0 in
# the rest, and no model ever split on it. The collector still writes the column
# and src/cams.py still fetches when credentials exist, so activating CAMS is
# a backfill plus putting "cams_pollen" back in this list and retraining.
CAMS_FEATURES: list[str] = []

# Intra-day relative features (capture diurnal position and short-term dynamics)
INTRADAY_FEATURES = [
    "temp_vs_daily_max",       # ratio of window temp to day's forecasted max (0-1+)
    "precip_in_prior_window",  # binary: did it rain in the preceding 3h window?
    "temp_rate_of_change",     # temperature change from previous 3h window (°C)
]

# Upwind stations (src/upwind.py): the highest reading at any of the ePIN
# stations around Munich over the 24 h and 7 d before the forecast origin, in
# log space, and how far the 24 h upwind maximum sits above Munich's own.
# Anchored at the origin for every lead, like the lag block. NaN where no
# station reported.
UPWIND_FEATURES = [
    "upwind_max_8",     # 24h max over upwind stations (log1p)
    "upwind_max_56",    # 7-day max over upwind stations (log1p)
    "upwind_lead_8",    # upwind_max_8 - pollen_max_8: upwind ahead of Munich
]

# How far ahead of the last measurement a prediction is being made, in 3h
# windows (1 = the first unforecast window). The lag block is anchored at the
# forecast origin rather than the target, so this is what tells the model how
# stale that block is. See src/rollout.py for why the model is direct rather
# than recursive.
LEAD_FEATURES = [
    "lead_windows",
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
    + LOAD_FEATURES
    + CAMS_FEATURES
    + INTRADAY_FEATURES
    + LAG_FEATURES
    + UPWIND_FEATURES
    + LEAD_FEATURES
)

# GDD base temperature (°C) — standard for temperate deciduous phenology
GDD_T_BASE = 5.0

# Species-specific GDD thresholds for pollen burst readiness (#2)
# When cumulative GDD exceeds this, species is primed for explosive release.
#
# These are a FALLBACK, not the live values. src/onset.py calibrates the
# threshold per species per year from the seasons before it, and that
# calibration is what the features actually use once two seasons of history
# exist. Measured against eight Munich seasons these constants cross 15–23 days
# after the observed onset, so they are kept only for a cold start.
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
    "Fungus":    (1, 12),   # Spores are in the air all year; the peak is summer–autumn
}


def is_season_active(species: str, month: int) -> bool:
    """Check if a species is within its **core** pollen season for a given month."""
    window = SPECIES_SEASON.get(species)
    if window is None:
        return True  # unknown species: assume always active
    start, end = window
    if start <= end:
        return start <= month <= end
    # wraps around year (e.g., Nov–Feb)
    return month >= start or month <= end


# Typical flowering-onset day-of-year per species (central-European baseline).
# Also a fallback: src/onset.py measures onset from the accumulated history and
# only falls back here for a species that has never been observed to start.
SPECIES_TYPICAL_ONSET_DOY: dict[str, int] = {
    "Corylus":   45,    # mid-February (hazel)
    "Alnus":     55,    # late February (alder)
    "Populus":   95,    # early April (poplar)
    "Salix":     95,    # early April (willow)
    "Fraxinus":  105,   # mid-April (ash)
    "Betula":    110,   # late April (birch)
    "Quercus":   125,   # early May (oak)
    "Poaceae":   145,   # late May (grasses)
    "Urtica":    160,   # June (nettle)
    "Artemisia": 205,   # late July (mugwort)
    "Ambrosia":  225,   # mid-August (ragweed)
}

# Allow the model to emit non-zero predictions this many months before/after
# the core season window. The hard month cutoff was forcing early-onset events
# (e.g. a warm-December hazel/alder bloom) to zero; the shoulder lets the GDD
# and burst features drive predictions in the transition period instead.
SEASON_SHOULDER_MONTHS = 1


def season_gate_active(species: str, month: int) -> bool:
    """Whether the model is *allowed* to emit a non-zero prediction this month.

    This is the **widened** season window (core ± SEASON_SHOULDER_MONTHS). It is
    used to decide when to force predictions to zero. The binary ``season_active``
    feature still uses the narrower :func:`is_season_active` core window so the
    model keeps learning the canonical season.
    """
    window = SPECIES_SEASON.get(species)
    if window is None:
        return True
    start, end = window
    s = start - SEASON_SHOULDER_MONTHS
    e = end + SEASON_SHOULDER_MONTHS
    # Expand into the explicit set of allowed months, wrapping around the year.
    allowed = {((s - 1 + i) % 12) + 1 for i in range(e - s + 1)}
    return month in allowed


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
    # No DWD index exists for spores. These are quantiles of the Munich
    # station's own daily means in the 2024+ sampler regime (median 46,
    # 85th percentile ~150, 99th ~300; see TASKS.md, F.1), so "moderate" is
    # an above-average day and "very high" a top-1% one.
    "Fungus":    (50, 150, 300),
}

# Default thresholds when species is unknown
_DEFAULT_THRESHOLDS = (10, 50, 200)


def value_to_level(value: float, species: str | None = None) -> PollenLevel:
    """Convert a numeric pollen value to a categorical level.

    When *species* is given, use DWD species-specific thresholds. Those
    thresholds are defined on **daily means**, so *value* must be one: pass a
    calendar day's mean, not a single 3h window (see :func:`daily_levels`).
    A midday window runs at two to three times its day's mean, and 15% of
    the pollen-bearing windows in the history read a level higher than
    their day when the thresholds are applied to them directly.
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


def daily_levels(
    frame: "Any", value: str, species: str = "species", by: tuple[str, ...] = ()
) -> "Any":
    """The level of each row's calendar day, from the day's mean of *value*.

    *frame* needs a ``date`` column; *by* names further columns a day is
    grouped by (the rollout groups by forecast origin as well, so each
    forecast's own daily mean is what gets levelled). Returns a Series
    aligned with *frame*.
    """
    import pandas as pd

    day = pd.to_datetime(frame["date"]).dt.normalize()
    means = frame.groupby([day, frame[species], *[frame[c] for c in by]])[value].transform("mean")
    return pd.Series(
        [value_to_level(float(v), str(sp)).value for v, sp in zip(means, frame[species])],
        index=frame.index,
    )


@dataclass
class SpeciesForecast:
    """Forecast for a single species in one time window.

    ``confidence`` is P(this level is exactly right) and
    ``confidence_within_one`` is P(the truth is within one level of it), both
    measured by the rollout benchmark rather than assumed — see
    :mod:`src.confidence`. ``value_low``/``value_high`` bound the central
    80% interval for ``value`` from the same benchmark's residuals (D.5);
    None when there is no interval to give.
    """
    name: str
    level: str
    value: float
    confidence: float
    confidence_within_one: float = 0.0
    value_low: float | None = None
    value_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        out: dict[str, Any] = {
            "name": self.name,
            "level": self.level,
            "value": round(self.value, 1),
            "confidence": round(self.confidence, 3),
            "confidence_within_one": round(self.confidence_within_one, 3),
        }
        if self.value_low is not None and self.value_high is not None:
            out["value_low"] = round(self.value_low, 1)
            out["value_high"] = round(self.value_high, 1)
        return out


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
class ObservationStatus:
    """How current the measurements behind a forecast are.

    ``age_windows`` counts the complete 3 h windows that have passed since the
    newest measurement without one — 0 or 1 on a normal run, since the station
    reports with a few hours' delay. ``stale`` is set once that reaches
    :data:`STALE_AFTER_WINDOWS`; the forecast then rests on data a day or
    more old, and its confidence is the confidence of the longer horizon it
    really is (see :func:`src.confidence.confidence_for`).
    """
    last: str | None
    age_windows: int | None
    stale: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {"last": self.last, "age_windows": self.age_windows, "stale": self.stale}


# A forecast whose newest measurement is this many complete windows old (a
# full day) is marked stale in the output.
STALE_AFTER_WINDOWS = 8


@dataclass
class RunStatus:
    """What one forecast run was built from: the inputs it did *not* get.

    ``observations`` is the newest measurement over all species, and
    ``species`` the same per species. ``defaulted`` names every input group
    that fell back to a default this run — NDVI, the DWD blend, the soil
    weather variables, the upwind stations, the calibration table, a species'
    model — with the reason, so the published forecast says which of its
    inputs were missing rather than looking the same as one that had them all.
    """
    observations: ObservationStatus
    species: dict[str, ObservationStatus] = field(default_factory=dict)
    defaulted: dict[str, str] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """True when anything was missing: stale observations or a default."""
        return self.observations.stale or bool(self.defaulted)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "degraded": self.degraded,
            "observations": {
                **self.observations.to_dict(),
                "species": {name: st.to_dict() for name, st in sorted(self.species.items())},
            },
            "defaulted": dict(sorted(self.defaulted.items())),
        }

    def describe(self) -> str:
        """One line for the run log."""
        obs = self.observations
        if obs.last is None:
            head = "no observations"
        else:
            head = f"newest observation {obs.last}, {obs.age_windows} window(s) old"
            if obs.stale:
                head += " (STALE)"
        if self.defaulted:
            tail = "; defaulted: " + ", ".join(
                f"{k} ({v})" for k, v in sorted(self.defaulted.items())
            )
        else:
            tail = "; all inputs present"
        return head + tail


@dataclass
class ForecastOutput:
    """Top-level forecast output with metadata and daily forecasts.

    ``status`` (D.2/D.3) says how old the observations were and which input
    groups were defaulted; it is published beside the forecast so a reader
    can tell a degraded run from a normal one.
    """
    generated: str
    location: str
    forecast: list[DayForecast] = field(default_factory=list)
    status: RunStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        out: dict[str, Any] = {
            "generated": self.generated,
            "location": self.location,
            "forecast": [d.to_dict() for d in self.forecast],
        }
        if self.status is not None:
            out["status"] = self.status.to_dict()
        return out

    def to_web_dict(self) -> dict[str, Any]:
        """Serialize to a webapp-compatible dict matching the LGL measurement format.

        Restructures from window-centric (date→window→species) to
        species-centric (species→windows) with unix timestamps in seconds,
        matching the format returned by the ePIN LGL Bayern API. Each point
        additionally carries the emitted level, the calibrated confidence
        pair and the value interval from :class:`SpeciesForecast`, which the
        measurement format has no slot for; the frontend reads them alongside
        ``value``. The run's ``status`` block (observation age, defaulted
        inputs) is top-level.
        """
        from datetime import datetime, timedelta
        from .clock import LOCAL_TZ

        tz = LOCAL_TZ
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
                    point: dict[str, Any] = {
                        "from": from_unix,
                        "to": to_unix,
                        "value": round(sp.value, 1),
                        "level": sp.level,
                        "confidence": round(sp.confidence, 3),
                        "confidence_within_one": round(sp.confidence_within_one, 3),
                    }
                    if sp.value_low is not None and sp.value_high is not None:
                        point["value_low"] = round(sp.value_low, 1)
                        point["value_high"] = round(sp.value_high, 1)
                    species_data.setdefault(sp.name, []).append(point)

        measurements = [
            {"polle": name, "location": self.location, "data": data}
            for name, data in sorted(species_data.items())
        ]

        out: dict[str, Any] = {
            "generated": self.generated,
            "location": self.location,
            "measurements": measurements,
        }
        if self.status is not None:
            out["status"] = self.status.to_dict()
        return out
