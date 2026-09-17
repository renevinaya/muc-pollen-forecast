"""Missing raw inputs reach XGBoost as missing, derived gaps as 0."""

import numpy as np
import pandas as pd

from src.trainer import NAN_PASSTHROUGH, ZERO_MEANS_MISSING, finalize_features
from src.types import FEATURE_COLS, LAG_FEATURES, UPWIND_FEATURES, WEATHER_FEATURES


def test_raw_gaps_stay_missing_and_derived_gaps_become_zero() -> None:
    frame = pd.DataFrame(
        {
            "temperature_mean": [np.nan, 4.0],
            "boundary_layer_height": [0.0, 300.0],
            "ndvi": [0.0, 0.4],
            "soil_moisture_mean": [0.0, 0.2],
            "upwind_max_8": [np.nan, 1.0],
            "temp_delta_1d": [np.nan, 0.5],
            "onset_anomaly": [np.nan, -0.2],
        }
    )
    out = finalize_features(frame)
    assert np.isnan(out.loc[0, "temperature_mean"])
    assert np.isnan(out.loc[0, "boundary_layer_height"]), "a stored 0.0 is a missing marker"
    assert np.isnan(out.loc[0, "ndvi"])
    assert np.isnan(out.loc[0, "soil_moisture_mean"])
    assert np.isnan(out.loc[0, "upwind_max_8"])
    assert out.loc[0, "temp_delta_1d"] == 0.0
    assert out.loc[0, "onset_anomaly"] == 0.0
    assert out.loc[1, "boundary_layer_height"] == 300.0
    assert not frame.isna().equals(out.isna()) and frame.loc[0, "ndvi"] == 0.0, "input untouched"


def test_passthrough_set_is_the_raw_inputs_only() -> None:
    assert ZERO_MEANS_MISSING <= NAN_PASSTHROUGH
    assert set(WEATHER_FEATURES) <= NAN_PASSTHROUGH
    assert set(UPWIND_FEATURES) <= NAN_PASSTHROUGH
    assert not (set(LAG_FEATURES) & NAN_PASSTHROUGH)
    assert NAN_PASSTHROUGH <= set(FEATURE_COLS)
