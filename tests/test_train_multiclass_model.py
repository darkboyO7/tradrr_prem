"""Unit tests for the balanced 3-class multiclass model training pipeline."""

from datetime import UTC, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from models.ml_classifier import (
    CLASS_BUY,
    CLASS_NEUTRAL,
    CLASS_SELL,
    LightGBMPredictor,
    REQUIRED_NUM_CLASSES,
)
from scripts.train_multiclass_model import (
    FEATURE_COLUMNS,
    build_labeled_features,
    compute_multiclass_brier,
)


@pytest.fixture
def sample_candle_datasets() -> dict[str, pd.DataFrame]:
    """Generate deterministic synthetic candle data with clear UP, DOWN, and FLAT regimes."""
    np.random.seed(42)
    n = 120
    now = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    timestamps = [now + timedelta(minutes=5 * i) for i in range(n)]

    # Section 1: Uptrend, Section 2: Downtrend, Section 3: Flat
    returns = np.concatenate([
        np.full(40, 0.002),
        np.full(40, -0.002),
        np.full(40, 0.00005),
    ])
    prices = 2500.0 * np.exp(np.cumsum(returns))

    opens = prices * 0.999
    highs = prices * 1.002
    lows = prices * 0.998
    closes = prices
    volumes = np.random.uniform(1000, 5000, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return {"NSE:TEST": df}


def test_build_labeled_features_contract(sample_candle_datasets: dict[str, pd.DataFrame]) -> None:
    X, y, fwd_returns = build_labeled_features(
        sample_candle_datasets,
        lookahead_bars=2,
        return_threshold=0.001,
    )

    assert not X.empty
    assert len(X) == len(y) == len(fwd_returns)
    assert set(FEATURE_COLUMNS).issubset(set(X.columns))

    # Assert all 3 classes are present
    classes_found = set(np.unique(y))
    assert classes_found == {CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY}


def test_multiclass_brier_calculation() -> None:
    # 1. Perfect confidence: probability 1.0 on true class -> Brier = 0.0
    probas_perfect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    y_true = np.array([0, 1, 2])
    brier_perf = compute_multiclass_brier(probas_perfect, y_true)
    assert brier_perf == pytest.approx(0.0, abs=1e-5)

    # 2. Uniform uninformative: probability 1/3 each -> Brier = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 0.6667
    probas_uniform = np.full((3, 3), 1.0 / 3.0)
    brier_unif = compute_multiclass_brier(probas_uniform, y_true)
    assert brier_unif == pytest.approx(0.666666, abs=1e-4)


def test_strict_multiclass_fitting(sample_candle_datasets: dict[str, pd.DataFrame]) -> None:
    X, y, _ = build_labeled_features(
        sample_candle_datasets,
        lookahead_bars=2,
        return_threshold=0.001,
    )

    predictor = LightGBMPredictor(
        model_id="test_multiclass_model",
        version="v_test_001",
        feature_names=list(X.columns),
        n_estimators=15,
        min_child_samples=3,
    )

    # Must fit cleanly under strict_multiclass=True
    predictor.fit(X, y, strict_multiclass=True)

    assert predictor.is_fitted is True
    assert predictor.is_fully_supported is True
    assert predictor.fitted_classes_ == [CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY]

    # Predict proba contract: shape (N, 3), sums to 1.0
    probas = predictor.predict_proba(X.iloc[:5])
    assert probas.shape == (5, 3)
    np.testing.assert_allclose(probas.sum(axis=1), np.ones(5), atol=1e-5)


def test_build_labeled_features_with_atr_deadband(sample_candle_datasets: dict[str, pd.DataFrame]) -> None:
    X, y, fwd_returns = build_labeled_features(
        sample_candle_datasets,
        lookahead_bars=2,
        use_atr_deadband=True,
        atr_multiplier=0.25,
    )

    assert not X.empty
    assert len(X) == len(y) == len(fwd_returns)
    assert "atr_14" in X.columns
    classes_found = set(np.unique(y))
    assert classes_found == {CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY}

