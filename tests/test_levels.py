"""Pollen levels are daily-mean levels: the thresholds are defined on daily means."""

import numpy as np
import pandas as pd

from src.forecaster import apply_day_blend, blend_day_with_dwd
from src.types import SPECIES_THRESHOLDS, daily_levels, value_to_level


def test_daily_levels_read_the_day_mean_not_the_window() -> None:
    low, moderate, _ = SPECIES_THRESHOLDS["Betula"]
    # One day: a midday spike above the moderate boundary, quiet otherwise.
    dates = pd.date_range("2026-04-10", periods=8, freq="3h")
    values = [0, 0, 5, moderate + 20, 30, 5, 0, 0]
    frame = pd.DataFrame({"date": dates, "species": "Betula", "value": values})
    levels = daily_levels(frame, "value")
    day_level = value_to_level(float(np.mean(values)), "Betula").value
    assert set(levels) == {day_level}
    assert day_level == "moderate"
    # Applied to the 3h values directly, the spike would read "high" and the
    # night "none" — the overstatement D.4 removes.
    assert value_to_level(moderate + 20, "Betula").value == "high"
    assert value_to_level(0, "Betula").value == "none"


def test_daily_levels_group_by_extra_keys_and_species() -> None:
    dates = pd.date_range("2026-04-10", periods=8, freq="3h")
    frame = pd.concat(
        [
            pd.DataFrame({"date": dates, "species": "Betula", "origin": "a", "value": 100.0}),
            pd.DataFrame({"date": dates, "species": "Betula", "origin": "b", "value": 0.0}),
            pd.DataFrame({"date": dates, "species": "Alnus", "origin": "a", "value": 5.0}),
        ],
        ignore_index=True,
    )
    levels = daily_levels(frame, "value", by=("origin",))
    assert set(levels[frame.origin == "a"][frame.species == "Betula"]) == {"very_high"} or \
        set(levels[(frame.origin == "a") & (frame.species == "Betula")]) == {"high"}
    assert set(levels[(frame.origin == "b") & (frame.species == "Betula")]) == {"none"}
    assert set(levels[frame.species == "Alnus"]) == {"low"}


def test_blend_moves_the_day_mean_one_step_toward_dwd() -> None:
    low, moderate, high = SPECIES_THRESHOLDS["Betula"]
    assert blend_day_with_dwd(low / 2, "Betula", dwd_num=1) is None  # already "low"
    up = blend_day_with_dwd(low / 2, "Betula", dwd_num=3)  # DWD says high: one step only
    assert up is not None and low / 2 < up < moderate
    down = blend_day_with_dwd(moderate + 10, "Betula", dwd_num=0)
    assert down is not None and down < moderate + 10
    from_zero = blend_day_with_dwd(0.0, "Betula", dwd_num=2)
    assert from_zero == low / 4  # half of the low band's midpoint


def test_day_blend_keeps_the_diurnal_shape() -> None:
    dts = pd.date_range("2026-04-10", periods=4, freq="3h")
    values = dict(zip(dts, [0.0, 10.0, 30.0, 0.0]))
    out = apply_day_blend(values, target_mean=20.0)
    assert np.isclose(np.mean(list(out.values())), 20.0)
    assert out[dts[2]] / out[dts[1]] == 3.0
    flat = apply_day_blend(dict(zip(dts, [0.0] * 4)), target_mean=2.5)
    assert set(flat.values()) == {2.5}
