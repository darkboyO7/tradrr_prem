"""Trading Intelligence Platform - Multiclass Classifier Contract Regression Tests.

Validates:
1. Training data containing all three classes (SELL=0, NEUTRAL=1, BUY=2).
2. Training data missing NEUTRAL.
3. Training data missing BUY or SELL.
4. Training data containing only one class.
5. Probability columns returned in the wrong order or permuted classes.
6. NaN, infinite, negative, and non-normalized probabilities.
7. Prediction-to-action threshold boundaries (confidence, margin, NO_TRADE).
8. Missing or stale market data inputs.
9. Model loading and prediction after application restart / serialization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from models.base import ModelPrediction, SignalAction
from models.ml_classifier import (
    CLASS_BUY,
    CLASS_NEUTRAL,
    CLASS_SELL,
    REQUIRED_NUM_CLASSES,
    SUPPORTED_CLASSES,
    LightGBMPredictor,
    validate_and_normalize_probabilities,
)


def make_dummy_features(n_samples: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible feature matrix."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "return_1": rng.normal(0, 0.01, n_samples),
        "log_return_5": rng.normal(0, 0.02, n_samples),
        "candle_range_pct": rng.uniform(0.005, 0.02, n_samples),
        "candle_body_ratio": rng.uniform(0.1, 0.9, n_samples),
        "rsi_14": rng.uniform(20, 80, n_samples),
        "macd_hist": rng.normal(0, 0.5, n_samples),
        "adx_14": rng.uniform(10, 50, n_samples),
        "atr_14": rng.uniform(1, 5, n_samples),
        "rvol_20": rng.uniform(0.5, 3.0, n_samples),
        "volatility_regime": rng.randint(0, 4, n_samples),
    })


# =========================================================================
# TEST 1: Training data containing all three classes
# =========================================================================

def test_multiclass_training_all_three_classes(tmp_path: Path):
    """Ensure standard 3-class training guarantees contract order: 0=SELL, 1=NEUTRAL, 2=BUY."""
    X = make_dummy_features(150, seed=1)
    y = np.array([CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY] * 50)

    model = LightGBMPredictor(n_estimators=15, max_depth=3)
    model.fit(X, y, strict_multiclass=True)

    assert model.is_fitted is True
    assert model.is_fully_supported is True
    assert set(model.fitted_classes_) == {CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY}

    probas = model.predict_proba(X.iloc[:10])
    assert probas.shape == (10, REQUIRED_NUM_CLASSES)
    assert np.all(probas >= 0.0)
    assert np.all(probas <= 1.0)
    assert np.all(np.isfinite(probas))
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)


# =========================================================================
# TEST 2: Training data missing NEUTRAL
# =========================================================================

def test_training_data_missing_neutral():
    """Ensure missing NEUTRAL class raises in strict mode and uses safe Bayesian prior non-strict."""
    X = make_dummy_features(100, seed=2)
    # Target only contains SELL (0) and BUY (2)
    y = np.array([CLASS_SELL, CLASS_BUY] * 50)

    # 1. Strict mode must raise ValueError
    strict_model = LightGBMPredictor(n_estimators=10)
    with pytest.raises(ValueError, match="Incomplete class distribution"):
        strict_model.fit(X, y, strict_multiclass=True)

    # 2. Non-strict mode must NOT assign 0.0 to missing class
    model = LightGBMPredictor(n_estimators=10)
    model.fit(X, y, strict_multiclass=False)

    assert model.is_fitted is True
    assert model.is_fully_supported is False
    assert model.metadata.parameters["unsupported_classes"] == [CLASS_NEUTRAL]

    probas = model.predict_proba(X.iloc[:5])
    assert probas.shape == (5, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
    # Neutral column (index 1) must retain prior uncertainty > 0.0, never silently 0.0
    assert np.all(probas[:, CLASS_NEUTRAL] > 0.01)


# =========================================================================
# TEST 3: Training data missing BUY or SELL
# =========================================================================

def test_training_data_missing_buy_or_sell():
    """Ensure missing BUY (or SELL) is never silently 0.0 and flags unsupported."""
    X = make_dummy_features(100, seed=3)
    # Target missing BUY (contains only SELL=0 and NEUTRAL=1)
    y_no_buy = np.array([CLASS_SELL, CLASS_NEUTRAL] * 50)

    model = LightGBMPredictor(n_estimators=10)
    model.fit(X, y_no_buy, strict_multiclass=False)

    assert model.is_fitted is True
    assert model.is_fully_supported is False
    assert CLASS_BUY in model.metadata.parameters["unsupported_classes"]

    probas = model.predict_proba(X.iloc[:5])
    assert probas.shape == (5, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
    # BUY column (index 2) must retain prior uncertainty > 0.0
    assert np.all(probas[:, CLASS_BUY] > 0.01)


# =========================================================================
# TEST 4: Training data containing only one class
# =========================================================================

def test_training_data_containing_only_one_class():
    """Single class training data must not fit directional model and fallback to uniform prior."""
    X = make_dummy_features(60, seed=4)
    y_mono = np.full(60, CLASS_BUY)

    # Strict raises
    model_strict = LightGBMPredictor(n_estimators=10)
    with pytest.raises(ValueError, match="Insufficient class diversity"):
        model_strict.fit(X, y_mono, strict_multiclass=True)

    # Non-strict returns unfitted with safe uniform fallback
    model = LightGBMPredictor(n_estimators=10)
    model.fit(X, y_mono, strict_multiclass=False)

    assert model.is_fitted is False
    assert model.is_fully_supported is False

    probas = model.predict_proba(X.iloc[:5])
    assert probas.shape == (5, 3)
    assert np.allclose(probas, 1.0 / 3.0)


# =========================================================================
# TEST 5: Probability columns returned in the wrong order
# =========================================================================

def test_probability_columns_wrong_order_reordered():
    """Verify that permuted internal classifier classes are reordered to [SELL=0, NEUTRAL=1, BUY=2]."""
    X = make_dummy_features(20, seed=5)
    model = LightGBMPredictor(n_estimators=10)
    model.is_fitted = True

    # Mock an internal classifier whose classes_ are permuted: [BUY=2, SELL=0, NEUTRAL=1]
    class MockPermutedClassifier:
        classes_ = np.array([2, 0, 1])

        def predict_proba(self, clean_X):
            # Output: col 0 = BUY (0.7), col 1 = SELL (0.2), col 2 = NEUTRAL (0.1)
            n = len(clean_X)
            return np.tile([0.7, 0.2, 0.1], (n, 1))

    model.classifier = MockPermutedClassifier()

    probas = model.predict_proba(X.iloc[:3])
    assert probas.shape == (3, 3)
    # Column 0 (SELL) should be 0.2
    assert np.allclose(probas[:, CLASS_SELL], 0.2)
    # Column 1 (NEUTRAL) should be 0.1
    assert np.allclose(probas[:, CLASS_NEUTRAL], 0.1)
    # Column 2 (BUY) should be 0.7
    assert np.allclose(probas[:, CLASS_BUY], 0.7)


# =========================================================================
# TEST 6: NaN, infinite, negative, and non-normalized probabilities
# =========================================================================

def test_nan_infinite_negative_probability_handling():
    """Directly test validation and normalization under malicious/corrupt probability inputs."""
    # 1. NaN and Inf
    corrupt = np.array([
        [np.nan, 0.5, 0.5],
        [np.inf, 0.2, -np.inf],
        [-0.5, 0.5, 1.5],
        [0.0, 0.0, 0.0],  # All zero row
        [10.0, 20.0, 30.0],  # Unnormalized row
    ])

    cleaned = validate_and_normalize_probabilities(corrupt, n_expected_rows=5)
    assert cleaned.shape == (5, 3)
    assert np.all(np.isfinite(cleaned))
    assert np.all(cleaned >= 0.0)
    assert np.all(cleaned <= 1.0)
    assert np.allclose(cleaned.sum(axis=1), 1.0)

    # All-zero row must safely resolve to uniform [1/3, 1/3, 1/3]
    assert np.allclose(cleaned[3], 1.0 / 3.0)

    # 2. Shape mismatch raises ValueError
    bad_shape = np.ones((5, 2))
    with pytest.raises(ValueError, match="Probability shape contract violation"):
        validate_and_normalize_probabilities(bad_shape, n_expected_rows=5)


# =========================================================================
# TEST 7: Prediction-to-action threshold boundaries
# =========================================================================

def test_prediction_to_action_threshold_boundaries():
    """Verify that NO_TRADE is emitted whenever evidence is insufficient, flat, or tied."""
    model = LightGBMPredictor(min_confidence=0.45, min_margin=0.05)
    model.is_fitted = True

    # Test distinct probability vectors
    class MockProbClassifier:
        classes_ = np.array([0, 1, 2])
        def __init__(self, prob_rows):
            self.prob_rows = np.array(prob_rows)
        def predict_proba(self, clean_X):
            return self.prob_rows

    test_cases = [
        # [p_sell, p_neutral, p_buy] -> expected_action
        ([0.20, 0.60, 0.20], SignalAction.NO_TRADE),  # Neutral highest
        ([0.35, 0.33, 0.32], SignalAction.NO_TRADE),  # Insufficient confidence (< 0.45)
        ([0.48, 0.04, 0.48], SignalAction.NO_TRADE),  # Tied BUY and SELL (margin < 0.05)
        ([0.46, 0.44, 0.10], SignalAction.NO_TRADE),  # SELL over NEUTRAL margin only 0.02 (< 0.05)
        ([0.10, 0.25, 0.65], SignalAction.BUY),       # Clear BUY: conf 0.65, margin 0.40
        ([0.70, 0.15, 0.15], SignalAction.SELL),      # Clear SELL: conf 0.70, margin 0.55
    ]

    for prob_vec, expected_act in test_cases:
        model.classifier = MockProbClassifier([prob_vec])
        dummy_df = pd.DataFrame({"return_1": [0.0]})
        preds = model.predict(dummy_df)
        assert len(preds) == 1
        assert preds[0].action == expected_act, f"Failed for {prob_vec}: expected {expected_act}, got {preds[0].action}"


# =========================================================================
# TEST 8: Missing or stale market data
# =========================================================================

def test_missing_or_stale_market_data():
    """Verify that empty, incomplete, or NaN feature DataFrames do not crash prediction."""
    X_train = make_dummy_features(100, seed=8)
    y_train = np.array([0, 1, 2] * 33 + [0])
    model = LightGBMPredictor(n_estimators=10)
    model.fit(X_train, y_train)

    # 1. Missing columns: DataFrame with completely unrelated columns
    missing_cols_df = pd.DataFrame({"unknown_col": [1.0, 2.0, 3.0]})
    probas = model.predict_proba(missing_cols_df)
    assert probas.shape == (3, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)

    # 2. DataFrame with NaNs and Infs across all feature cells
    corrupt_df = pd.DataFrame({
        feat: [np.nan, np.inf, -np.inf] for feat in model.metadata.feature_names
    })
    probas_corrupt = model.predict_proba(corrupt_df)
    assert probas_corrupt.shape == (3, 3)
    assert np.all(np.isfinite(probas_corrupt))
    assert np.allclose(probas_corrupt.sum(axis=1), 1.0)

    preds = model.predict(corrupt_df)
    assert len(preds) == 3


# =========================================================================
# TEST 9: Model loading and prediction after application restart
# =========================================================================

def test_model_loading_and_prediction_after_restart(tmp_path: Path):
    """Verify round-trip serialization preserves multiclass contract, weights, and thresholds."""
    X_train = make_dummy_features(120, seed=9)
    y_train = np.array([0, 1, 2] * 40)

    model = LightGBMPredictor(
        model_id="test_restart_lgbm",
        version="v2.1.0",
        n_estimators=15,
        min_confidence=0.46,
        min_margin=0.06,
    )
    model.fit(X_train, y_train, strict_multiclass=True)

    test_slice = X_train.iloc[:5]
    probas_before = model.predict_proba(test_slice)
    preds_before = [p.action for p in model.predict(test_slice)]

    # Save to disk
    save_dir = tmp_path / "saved_restart_model"
    model.save(save_dir)

    # Deserialize in a clean fresh instance
    loaded_model = LightGBMPredictor.load(save_dir)

    assert loaded_model.is_fitted is True
    assert loaded_model.is_fully_supported is True
    assert loaded_model.min_confidence == 0.46
    assert loaded_model.min_margin == 0.06

    probas_after = loaded_model.predict_proba(test_slice)
    preds_after = [p.action for p in loaded_model.predict(test_slice)]

    assert np.allclose(probas_before, probas_after, atol=1e-6)
    assert preds_before == preds_after
