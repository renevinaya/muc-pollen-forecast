"""Tests for measured forecast confidence.

The published confidence used to be an invented decay. What matters now is that
it stays tied to something measured, that it degrades safely when the
measurement is missing, and that it does not quietly regrow the false precision
the leave-one-fold-out comparison ruled out.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.confidence import (
    EMIT_THRESHOLD,
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    NO_MODEL_SCALE,
    OBSERVED_CONFIDENCE,
    TABLE_PATH,
    build_table,
    confidence_for,
    expected_calibration_error,
    load_table,
)

LEVELS = ["low", "moderate", "high", "very_high"]


def synthetic_results(n: int = 4000, accuracy: float = 0.35, seed: int = 3) -> pd.DataFrame:
    """Benchmark-shaped results with a known exact-match rate."""
    rng = np.random.default_rng(seed)
    predicted = rng.choice(LEVELS, n)
    hit = rng.random(n) < accuracy
    actual = [
        p if h else rng.choice([lv for lv in LEVELS if lv != p])
        for p, h in zip(predicted, hit)
    ]
    return pd.DataFrame({
        "species": rng.choice(["Betula", "Alnus", "Poaceae"], n),
        "horizon_day": rng.integers(1, 6, n),
        "fold": rng.integers(1, 7, n),
        "origin": rng.integers(0, 50, n),
        "predicted": rng.uniform(1.0, 50.0, n),
        "level_predicted": predicted,
        "level_actual": actual,
    })


def test_table_recovers_the_true_accuracy() -> None:
    table = build_table(synthetic_results(accuracy=0.35))
    assert table["overall"]["exact"] == pytest.approx(0.35, abs=0.03)


def test_only_emitted_rows_are_calibrated() -> None:
    """Rows below the emit threshold never reach a user, so they must not count.

    Pooled over everything the model looks far more accurate than it is,
    because `none` predictions dominate and are easy.
    """
    results = synthetic_results(accuracy=0.35)
    dropped = results.copy()
    dropped["predicted"] = EMIT_THRESHOLD / 2          # nothing would be emitted
    dropped["level_actual"] = dropped["level_predicted"]  # ...and all "correct"

    table = build_table(pd.concat([results, dropped], ignore_index=True))
    assert table["overall"]["exact"] == pytest.approx(0.35, abs=0.03)


def test_within_one_is_never_below_exact() -> None:
    table = build_table(synthetic_results())
    assert table["overall"]["within_one"] >= table["overall"]["exact"]


def test_observed_windows_are_certain() -> None:
    """An assimilated window carries a measurement, not a prediction."""
    table = build_table(synthetic_results())
    exact, within = confidence_for(table, "Betula", "high", 3, observed=True)
    assert (exact, within) == (OBSERVED_CONFIDENCE["exact"], OBSERVED_CONFIDENCE["within_one"])


def test_missing_table_still_publishes_something_sane() -> None:
    """A missing table must not resurrect an optimistic default."""
    exact, within = confidence_for(None, "Betula", "high", 1)
    assert MIN_CONFIDENCE <= exact <= MAX_CONFIDENCE
    assert exact < 0.6, "fallback confidence must not overstate a ~35% reality"
    assert within >= exact


def test_no_model_species_is_marked_down() -> None:
    table = build_table(synthetic_results())
    with_model, _ = confidence_for(table, "Betula", "low", 1, has_model=True)
    without, _ = confidence_for(table, "Betula", "low", 1, has_model=False)
    assert without == pytest.approx(with_model * NO_MODEL_SCALE, abs=1e-6)


def test_confidence_is_not_keyed_on_species_or_level() -> None:
    """Fine-grained keys calibrate worse held-out, so they must stay unused.

    Asserted rather than merely documented: the natural instinct is to key on
    them, and doing so silently reintroduces false precision.
    """
    table = build_table(synthetic_results())
    baseline = confidence_for(table, "Betula", "low", 2)
    for species in ("Alnus", "Poaceae", "Nonexistent"):
        for level in LEVELS:
            assert confidence_for(table, species, level, 2) == baseline


def test_confidence_varies_with_horizon_only() -> None:
    table = build_table(synthetic_results())
    values = {confidence_for(table, "Betula", "low", h)[0] for h in range(1, 6)}
    assert len(values) > 1, "horizon offset should still move the number"
    assert max(values) - min(values) < 0.1, "horizon effect is measured as small"


def test_ece_rewards_a_calibrated_scheme() -> None:
    """The metric must prefer a truthful number to a flattering one."""
    correct = pd.Series([True] * 350 + [False] * 650)
    honest = pd.Series([0.35] * 1000)
    flattering = pd.Series([0.90] * 1000)
    assert expected_calibration_error(honest, correct) < 0.05
    assert expected_calibration_error(flattering, correct) > 0.5


def test_shipped_table_is_present_and_current() -> None:
    """The committed table must exist and describe a plausible model."""
    table = load_table()
    assert table is not None, f"{TABLE_PATH} is missing — run 'calibrate'"
    assert 0.15 < table["overall"]["exact"] < 0.75
    assert table["overall"]["within_one"] > table["overall"]["exact"]
    assert set(table["horizon_delta"]) == {"1", "2", "3", "4", "5"}
    assert table["source"]["rows_emitted"] > 1000


def test_shipped_table_is_valid_json() -> None:
    with TABLE_PATH.open() as handle:
        json.load(handle)
