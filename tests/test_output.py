"""The published forecast.json carries the level and confidence per point."""

import json

from src.types import DayForecast, ForecastOutput, SpeciesForecast, WindowForecast


def _output() -> ForecastOutput:
    window = WindowForecast(
        from_time="06:00",
        to_time="09:00",
        species=[
            SpeciesForecast("Betula", "high", 120.4, 0.41, 0.9, value_low=30.2, value_high=250.0),
            SpeciesForecast("Alnus", "low", 3.0, 0.3, 0.85),
        ],
    )
    return ForecastOutput(
        generated="2026-04-01T05:00:00Z",
        location="DEMUNC",
        forecast=[DayForecast(date="2026-04-01", windows=[window])],
    )


def test_web_dict_points_carry_level_and_confidence() -> None:
    web = _output().to_web_dict()
    by_name = {m["polle"]: m for m in web["measurements"]}
    assert set(by_name) == {"Alnus", "Betula"}
    point = by_name["Betula"]["data"][0]
    assert point["value"] == 120.4
    assert point["level"] == "high"
    assert point["confidence"] == 0.41
    assert point["confidence_within_one"] == 0.9
    assert (point["value_low"], point["value_high"]) == (30.2, 250.0)
    # A point without an interval simply has no interval keys.
    assert "value_low" not in by_name["Alnus"]["data"][0]
    assert point["to"] - point["from"] == 3 * 3600
    assert by_name["Betula"]["location"] == "DEMUNC"


def test_web_dict_is_json_serialisable() -> None:
    text = json.dumps(_output().to_web_dict())
    assert '"confidence_within_one": 0.85' in text
