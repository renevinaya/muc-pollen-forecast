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
from tests.test_feature_parity import SPECIES, build_history, build_upwind

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


# Features the synthetic fixture cannot make vary. Empty on purpose: an
# exemption here hides exactly what the test is for, so anything added needs
# the constant-feature check run against the real history first.
FIXTURE_CANNOT_TRIGGER: set[str] = set()


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    # Three completed seasons: the interannual load features are 0 by design
    # until a species has a previous season *and* a baseline before that.
    return build_history(days=1100, seed=5, species_list=[SPECIES])


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
    x, _, _ = prepare_training_data(history, SPECIES, upwind=build_upwind(history, SPECIES))
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


def test_retired_columns_are_not_model_inputs_or_derivation_inputs() -> None:
    """A column may only be retired if nothing reads it."""
    from src.types import RETIRED_COLUMNS

    assert not set(RETIRED_COLUMNS) & set(WEATHER_COLUMNS)
    assert not set(RETIRED_COLUMNS) & set(DERIVATION_INPUTS)
    # The calendar features are still model inputs — derived, not stored.
    stored_inputs = set(RETIRED_COLUMNS) & set(FEATURE_COLS)
    assert stored_inputs == {
        "day_of_year_sin", "day_of_year_cos", "hour_of_day", "hour_sin", "hour_cos",
    }


def test_writing_the_history_drops_the_retired_columns(tmp_path, monkeypatch) -> None:
    import src.collector as collector
    from src.types import RETIRED_COLUMNS

    monkeypatch.setattr(collector, "DATA_DIR", tmp_path)
    monkeypatch.setattr(collector, "HISTORY_FILE", tmp_path / "history.csv")
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-10 03:00"]), "species": ["Betula"], "value": [0.0],
        "temperature_mean": [4.2], "cape_max": [float("nan")], "evi": [0.3],
        "hour_sin": [0.7], "cams_pollen": [0.0],
    })
    written = collector.write_history(frame)
    assert list(written.columns) == ["date", "species", "value", "temperature_mean"]
    stored = pd.read_csv(tmp_path / "history.csv")
    assert not set(stored.columns) & set(RETIRED_COLUMNS)


def test_calendar_features_do_not_depend_on_stored_columns(history: pd.DataFrame) -> None:
    """Training derives them from the timestamp, whatever the history carries."""
    poisoned = history.assign(hour_of_day=-1.0, hour_sin=9.0, day_of_year_sin=9.0)
    x, _, _ = prepare_training_data(poisoned, SPECIES, leads=(1,))
    assert set(x["hour_of_day"].unique()) == {0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0}
    assert x["hour_sin"].abs().max() <= 1.0 and x["day_of_year_sin"].abs().max() <= 1.0
