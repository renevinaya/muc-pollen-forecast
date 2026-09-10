"""Tests for the stage-3 extreme-regressor blend and its gate.

The extreme regressor is fitted only to samples above ``extreme_threshold``, so
it has never seen an ordinary window and its opinion about one is worthless.
Everything here is about making sure it is only consulted when the window is
actually likely to be extreme — and that models pickled before the gate existed
still load and are served safely.
"""

import numpy as np
import pandas as pd
import pytest

from src.trainer import (
    EXTREME_GATE_HI,
    EXTREME_GATE_LO,
    EXTREME_MAX_WEIGHT,
    TwoStageModel,
    prepare_training_data,
    train_species_model,
)
from src.types import FEATURE_COLS
from tests.test_feature_parity import SPECIES, build_history


@pytest.fixture(scope="module")
def trained() -> tuple[TwoStageModel, pd.DataFrame, pd.Series]:
    history = build_history(days=500, seed=11, species_list=[SPECIES])
    x, y, raw = prepare_training_data(history, SPECIES)
    model = train_species_model(x, y, raw_values=raw, species=SPECIES)
    assert model is not None
    return model, x, raw


def test_gate_classifier_is_trained(trained) -> None:
    model, _, raw = trained
    assert (raw > model.extreme_threshold).sum() >= 10, "fixture has no extreme samples"
    assert model.extreme_regressor is not None
    assert model.extreme_classifier is not None


def test_gate_predicts_the_right_event(trained) -> None:
    """The gate must be P(value > threshold), not P(value > 0).

    Those two coincided for weeks at a time in peak season, which is what made
    the old gate fire on ordinary windows.
    """
    model, x, raw = trained
    prob_extreme = model.extreme_classifier.predict_proba(x)[:, 1]
    is_extreme = (raw.to_numpy(dtype=float) > model.extreme_threshold)

    assert prob_extreme[is_extreme].mean() > prob_extreme[~is_extreme].mean()
    # And it must be meaningfully different from the P(>0) classifier it replaced.
    prob_active = model.classifier.predict_proba(x)[:, 1]
    active_only = (raw.to_numpy(dtype=float) > 0) & ~is_extreme
    assert prob_extreme[active_only].mean() < prob_active[active_only].mean()


def test_blend_is_rare_on_ordinary_windows(trained) -> None:
    """The blend must not fire across a fifth of the record.

    Under the old gate it fired on 20–28% of all windows on the real history,
    with the truth at or below the threshold about three quarters of the time.
    """
    model, x, raw = trained
    prob_extreme = model.extreme_classifier.predict_proba(x)[:, 1]
    gate = np.clip(
        (prob_extreme - EXTREME_GATE_LO) / (EXTREME_GATE_HI - EXTREME_GATE_LO), 0.0, 1.0
    )
    fires = gate > 0
    rv = raw.to_numpy(dtype=float)

    assert fires.mean() < 0.15, f"blend fires on {fires.mean():.1%} of windows"
    if fires.any():
        # When it does fire, it should mostly be on genuinely large values.
        assert (rv[fires] > model.extreme_threshold).mean() > 0.5


def test_zero_windows_are_never_blended(trained) -> None:
    """A window whose truth is zero must not be handed the extreme model."""
    model, x, raw = trained
    rv = raw.to_numpy(dtype=float)
    zero = rv == 0
    prob_extreme = model.extreme_classifier.predict_proba(x)[:, 1]
    assert prob_extreme[zero].max() < EXTREME_GATE_HI


def test_legacy_model_without_gate_still_predicts(trained) -> None:
    """Models pickled before the gate existed must keep working.

    The 3-hourly pipeline picks up new code immediately but keeps serving the
    models on the data release until the next retrain, so this combination runs
    in production.
    """
    model, x, _ = trained
    legacy = TwoStageModel(
        classifier=model.classifier,
        regressor=model.regressor,
        extreme_regressor=model.extreme_regressor,
        species=model.species,
    )
    del legacy.extreme_classifier  # exactly what unpickling an old dataclass gives

    preds = legacy.predict(x[FEATURE_COLS].head(64))
    assert len(preds) == 64
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0)


def test_legacy_model_skips_the_blend_entirely(trained) -> None:
    """Without a gate the blend is skipped, not run on the old broken one."""
    model, x, _ = trained
    sample = x[FEATURE_COLS].head(256)

    legacy = TwoStageModel(
        classifier=model.classifier,
        regressor=model.regressor,
        extreme_regressor=model.extreme_regressor,
        species=model.species,
    )
    del legacy.extreme_classifier

    no_stage3 = TwoStageModel(
        classifier=model.classifier,
        regressor=model.regressor,
        extreme_regressor=None,
        species=model.species,
    )
    assert np.allclose(legacy.predict(sample), no_stage3.predict(sample))


def test_blend_weight_never_exceeds_its_cap(trained) -> None:
    """Stage 2 always retains at least 1 - EXTREME_MAX_WEIGHT of the answer."""
    model, x, _ = trained
    sample = x[FEATURE_COLS].head(512)

    blended = model.predict(sample)
    without = TwoStageModel(
        classifier=model.classifier,
        regressor=model.regressor,
        extreme_regressor=None,
        species=model.species,
    ).predict(sample)
    extreme = np.maximum(0.0, model.extreme_regressor.predict(sample))

    # blended is a convex combination of `without` and `extreme`, capped.
    lo = np.minimum(without, extreme) - 1e-9
    hi = np.maximum(without, extreme) + 1e-9
    assert np.all((blended >= lo) & (blended <= hi))

    moved = np.abs(blended - without)
    span = np.abs(extreme - without)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(span > 1e-9, moved / span, 0.0)
    assert share.max() <= EXTREME_MAX_WEIGHT + 1e-6


def test_model_records_the_features_it_was_fitted_on(trained) -> None:
    model, x, _ = trained
    assert model.feature_names == list(x[FEATURE_COLS].columns)


def test_load_models_refuses_a_stale_feature_set(trained, tmp_path, monkeypatch) -> None:
    """A feature-set change must not take the live forecast down.

    The pipeline checks out new code every run but keeps serving the release's
    models until the next retrain, so a commit that adds a feature is briefly
    live against models fitted without it. XGBoost raises on that mismatch.
    """
    import joblib

    from src import trainer

    model, _, _ = trained
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)

    model.feature_names = ["some", "older", "feature", "set"]
    joblib.dump(model, tmp_path / f"{SPECIES}.joblib")

    assert SPECIES not in trainer.load_models()

    model.feature_names = list(FEATURE_COLS)
    joblib.dump(model, tmp_path / f"{SPECIES}.joblib")
    assert SPECIES in trainer.load_models()


def test_load_models_checks_legacy_models_by_feature_count(trained, tmp_path, monkeypatch) -> None:
    """Models pickled before feature_names existed are judged on shape instead."""
    import joblib

    from src import trainer

    model, _, _ = trained
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)

    model.feature_names = None  # what unpickling a pre-field model gives
    joblib.dump(model, tmp_path / f"{SPECIES}.joblib")
    # This model really was fitted on the current set, so shape agrees.
    assert SPECIES in trainer.load_models()
