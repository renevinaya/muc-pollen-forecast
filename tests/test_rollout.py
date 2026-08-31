"""Tests for the autoregressive rollout backtest.

The rollout exists to stop the benchmark flattering the product, so the
properties worth pinning are the ones that would quietly restore the flattery:
that predictions really are fed back into the lags, that the model is never
trained on the fold it is scored on, and that the horizon labelling lines up
with the windows it claims to describe.
"""

import numpy as np
import pandas as pd
import pytest

from src.rollout import eligible_months, rollout_evaluate
from src.types import WINDOWS_PER_DAY
from tests.test_feature_parity import SPECIES, build_history


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    # Two years, so a fold in the second one has a full season behind it.
    first = build_history(days=400, seed=3, species_list=[SPECIES])
    second = first.copy()
    second["date"] = second["date"] + pd.DateOffset(years=1)
    for col in ("day_of_year", "month", "hour_of_day"):
        second[col] = getattr(pd.DatetimeIndex(second["date"]), {
            "day_of_year": "dayofyear", "month": "month", "hour_of_day": "hour",
        }[col]).astype(float)
    return pd.concat([first, second], ignore_index=True)


@pytest.fixture(scope="module")
def results(history: pd.DataFrame) -> pd.DataFrame:
    return rollout_evaluate(
        history, horizon_days=3, months=[pd.Period("2022-04", "M")], species=[SPECIES]
    )


def test_produces_results(results: pd.DataFrame) -> None:
    assert not results.empty
    assert set(results["horizon_day"]) == {1, 2, 3}


def test_horizon_day_matches_offset_from_origin(results: pd.DataFrame) -> None:
    """horizon_day must describe the window it is attached to.

    An off-by-one here would silently attribute day-2 errors to day 1, which is
    precisely the overstatement the rollout was built to remove.
    """
    offset_days = (
        (results["date"] - results["origin"]) // pd.Timedelta(hours=3)
    ) // WINDOWS_PER_DAY + 1
    assert (offset_days == results["horizon_day"]).all()


def test_every_scored_window_is_after_its_origin(results: pd.DataFrame) -> None:
    assert (results["date"] >= results["origin"]).all()


def test_model_never_sees_the_fold_it_is_scored_on(history: pd.DataFrame) -> None:
    """Training data is strictly before the fold.

    Corrupting the fold's own measurements must not change the predictions: if
    it does, the fold leaked into training.
    """
    month = [pd.Period("2022-04", "M")]
    baseline = rollout_evaluate(history, horizon_days=1, months=month, species=[SPECIES])

    poisoned = history.copy()
    in_fold = (poisoned["date"] >= "2022-04-01") & (poisoned["date"] < "2022-05-01")
    poisoned.loc[in_fold & (poisoned["species"] == SPECIES), "value"] = 9999.0
    after = rollout_evaluate(poisoned, horizon_days=1, months=month, species=[SPECIES])

    merged = baseline.merge(after, on=["origin", "date"], suffixes=("_a", "_b"))
    assert not merged.empty
    # The first window of each origin still reads its lags from before the
    # origin, so only windows at or after the origin may move.
    first_windows = merged[merged["date"] == merged["origin"]]
    assert np.allclose(
        first_windows["predicted_a"], first_windows["predicted_b"], rtol=1e-9
    ), "the fold's own measurements changed a prediction made before them"


def test_predictions_feed_the_next_windows_lags(history: pd.DataFrame) -> None:
    """The whole point: later windows must depend on earlier predictions.

    With measured lags every window would be independent of the ones before it,
    which is the failure mode this module replaces. A rollout that produced a
    constant per origin would also pass a weaker check, so variation *within*
    an origin is what is asserted.
    """
    res = rollout_evaluate(
        history, horizon_days=3, months=[pd.Period("2022-04", "M")], species=[SPECIES]
    )
    spread = res.groupby("origin")["predicted"].nunique()
    assert (spread > 1).any(), "predictions never change across a forecast"


def test_eligible_months_requires_training_history(history: pd.DataFrame) -> None:
    """The earliest months can never be folds — nothing precedes them."""
    months = eligible_months(history)
    assert months
    earliest = pd.DatetimeIndex(history["date"]).to_period("M").min()
    assert earliest not in months
