"""Guards on the feature lists themselves.

Pruning a feature is a one-line edit to src/types.py, which makes it easy and
makes the failure modes quiet. These pin the two that would not raise:

* a raw column removed from what the pipeline *carries*, so a derived feature
  that reads it silently degrades to a constant;
* a feature left in FEATURE_COLS that nothing supplies, so it is served as 0.
"""

import pandas as pd
import pytest

from src.features import combined_weather
from src.trainer import (
    FEATURE_FAMILIES,
    _add_weather_derived_features,
    prepare_training_data,
)
from src.types import FEATURE_COLS, WEATHER_COLUMNS, WEATHER_FEATURES
from tests.test_feature_parity import SPECIES, build_history

# Raw columns that exist so a *derived* feature can be computed from them.
# Each is read by name in _add_weather_derived_features.
DERIVATION_INPUTS = [
    "temperature_mean",
    "sunshine_duration",
    "precipitation_sum",
    "humidity_mean",
    "wind_speed_max",
    "wind_direction",
]


# Rare-event flags the synthetic fixture never happens to produce. This is a
# limit of the fixture, not of the feature: on the real history
# cold_to_warm_flip fires on 2.1% of rows, and no feature in FEATURE_COLS is
# constant there. Anything added here needs that check run against real data
# first, otherwise the exemption hides exactly what the test is for.
FIXTURE_CANNOT_TRIGGER = {"cold_to_warm_flip"}


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return build_history(days=430, seed=5, species_list=[SPECIES])


def test_derivation_inputs_are_carried() -> None:
    """Every column a derived feature reads must survive into the context.

    WEATHER_FEATURES is the model's input list and WEATHER_COLUMNS is what the
    pipeline carries; they are deliberately different. Dropping `wind_direction`
    from the carried set does not raise — _add_weather_derived_features falls
    back to a constant 0 — it just quietly turns wind_dir_sin/cos and
    transport_south into constants.
    """
    missing = [c for c in DERIVATION_INPUTS if c not in WEATHER_COLUMNS]
    assert not missing, f"derived features read columns that are not carried: {missing}"


def test_wind_derivations_are_not_constant(history: pd.DataFrame) -> None:
    """The concrete symptom the test above is guarding against."""
    carried = combined_weather(history, pd.DataFrame())
    base = carried.reset_index().rename(columns={"index": "date"})
    base["species"] = SPECIES
    base["value"] = 0.0
    derived = _add_weather_derived_features(base, SPECIES)

    for col in ("wind_dir_sin", "wind_dir_cos", "transport_south"):
        assert derived[col].nunique() > 10, f"{col} collapsed to a constant"


def test_every_model_feature_is_actually_supplied(history: pd.DataFrame) -> None:
    """No feature may be in FEATURE_COLS without something producing it.

    A feature nothing fills is served as a constant 0 and looks exactly like a
    feature the model chose to ignore. `cams_pollen` sat in the list that way
    for its whole life.
    """
    x, _, _ = prepare_training_data(history, SPECIES)
    assert not x.empty
    constant = [
        c for c in FEATURE_COLS
        if c not in FIXTURE_CANNOT_TRIGGER and x[c].nunique(dropna=False) <= 1
    ]
    assert not constant, f"features that never vary in training: {constant}"


def test_feature_cols_has_no_duplicates() -> None:
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_gain_report_covers_every_feature() -> None:
    """The gain report groups by family, so a new family must be registered.

    `lead_windows` was reported under 'other' when it was added, which is how a
    feature ends up excluded from the evidence used to prune it.
    """
    grouped = {col for cols in FEATURE_FAMILIES.values() for col in cols}
    assert not set(FEATURE_COLS) - grouped, (
        f"features missing from FEATURE_FAMILIES: {set(FEATURE_COLS) - grouped}"
    )


def test_model_features_are_a_subset_of_carried_columns() -> None:
    assert not set(WEATHER_FEATURES) - set(WEATHER_COLUMNS)
