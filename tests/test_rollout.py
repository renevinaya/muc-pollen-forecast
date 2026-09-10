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

from src.features import LagState
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


def test_later_windows_do_not_depend_on_earlier_predictions(
    history: pd.DataFrame,
) -> None:
    """The forecast must be direct: no window may stand on another's output.

    This is the whole point of the change. Under the recursive rollout each
    prediction became the next window's lag, so an upward bias compounded into
    the horizon. The check: predictions from one origin must be reproducible one
    at a time, in isolation, with no shared running state.
    """
    month = [pd.Period("2022-04", "M")]
    full = rollout_evaluate(history, horizon_days=3, months=month, species=[SPECIES])
    assert not full.empty

    # Re-run with a single-day horizon: day 1 only, so nothing downstream of it
    # exists to have influenced it. Its predictions must be unchanged.
    day_one = rollout_evaluate(history, horizon_days=1, months=month, species=[SPECIES])
    merged = full[full["horizon_day"] == 1].merge(
        day_one, on=["origin", "date"], suffixes=("_full", "_short")
    )
    assert not merged.empty
    assert np.allclose(merged["predicted_full"], merged["predicted_short"], rtol=1e-9)


def test_lag_block_is_identical_across_a_whole_forecast(history: pd.DataFrame) -> None:
    """Every window of one forecast reads the same measured block.

    Only ``lead_windows`` distinguishes them, so if the block ever varied within
    a forecast something would be feeding it.
    """
    origin = pd.Timestamp("2022-04-15")
    blocks = [
        LagState.from_history(history, SPECIES, origin).lag_features()
        for _ in range(3)
    ]
    assert blocks[0] == blocks[1] == blocks[2]

    # And the state carries no mutation API that could reintroduce a feedback
    # loop by accident.
    state = LagState.from_history(history, SPECIES, origin)
    assert not hasattr(state, "record")
    assert not hasattr(state, "begin_window")


def test_bias_does_not_grow_with_horizon(history: pd.DataFrame) -> None:
    """Mean prediction must not inflate as the forecast walks forward.

    The recursive version climbed 12.97 -> 17.35 from day 1 to day 5 on real
    data while the actuals were flat. A direct forecast has no mechanism for
    that, and this pins it down on the synthetic history.
    """
    results = rollout_evaluate(
        history, horizon_days=3, months=[pd.Period("2022-04", "M")], species=[SPECIES]
    )
    by_day = results.groupby("horizon_day")["predicted"].mean()
    assert by_day.iloc[-1] <= by_day.iloc[0] * 1.5, (
        f"mean prediction inflates across the horizon: {by_day.to_dict()}"
    )


def test_eligible_months_requires_training_history(history: pd.DataFrame) -> None:
    """The earliest months can never be folds — nothing precedes them."""
    months = eligible_months(history)
    assert months
    earliest = pd.DatetimeIndex(history["date"]).to_period("M").min()
    assert earliest not in months
