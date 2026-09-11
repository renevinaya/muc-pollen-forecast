"""The retrain must refuse a feature that is mostly missing.

Three carried-but-empty columns have now been found by hand
(``days_since_active``, ``cams_pollen``, then the diurnal/soil/NDVI block).
Each one reached the model as zeros because ``prepare_training_data`` fills
NaN with 0, so nothing downstream ever complained. This guard is where it
complains.
"""

import numpy as np
import pandas as pd
import pytest

from src.trainer import FeatureCoverageError, check_feature_coverage, feature_coverage
from src.types import FEATURE_COLS, WEATHER_COLUMNS


def build_history(days: int = 20) -> pd.DataFrame:
    windows = pd.date_range("2024-01-01", periods=days * 8, freq="3h")
    rows = []
    for dt in windows:
        row = {"date": dt, "species": "Betula", "value": 1.0}
        for col in WEATHER_COLUMNS:
            row[col] = 3.0
        row.update({"ndvi": 0.5, "evi": 0.4, "ndvi_delta": 0.01})
        rows.append(row)
    return pd.DataFrame(rows)


def test_complete_history_passes():
    report = check_feature_coverage(build_history())
    assert (report["missing"] == 0).all()
    assert set(report["feature"]) <= set(FEATURE_COLS)


def test_nan_on_most_rows_is_refused():
    history = build_history()
    history.loc[history.index[: int(len(history) * 0.7)], "soil_temperature_mean"] = np.nan
    with pytest.raises(FeatureCoverageError, match="soil_temperature_mean"):
        check_feature_coverage(history)


def test_zero_counts_as_missing_only_where_zero_is_a_fill():
    history = build_history()
    history["ndvi"] = 0.0                # NDVI is never really 0 over Munich
    history["precipitation_sum"] = 0.0   # dry weather is real
    report = feature_coverage(history).set_index("feature")
    assert report.loc["ndvi", "missing"] == 1.0
    assert report.loc["precipitation_sum", "missing"] == 0.0
    with pytest.raises(FeatureCoverageError, match="ndvi"):
        check_feature_coverage(history)


def test_nan_on_a_minority_of_rows_is_reported_not_refused():
    history = build_history()
    history.loc[history.index[:10], "dew_point_mean"] = np.nan
    report = check_feature_coverage(history).set_index("feature")
    assert 0 < report.loc["dew_point_mean", "missing"] < 0.5
    assert report.loc["dew_point_mean", "first_present"] == history["date"].iloc[10]
