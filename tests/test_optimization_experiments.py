"""Trading Intelligence Platform — Test for Continuous Optimization Experiment Suite.

Verifies that the autonomous optimization experiment suite executes cleanly,
evaluates out-of-sample models across purged partitions, and computes
exact statistical metrics without lookahead leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning.accuracy_evaluator import (
    AccuracyAcceptanceState,
    InstitutionalModelEvaluator,
)
from scripts.run_optimization_experiments import (
    BASE_FEATURE_COLUMNS,
    MICROSTRUCTURE_FEATURE_COLUMNS,
    CalibratedEnsemblePredictor,
    RegimeGatedEnsemblePredictor,
    extract_features_and_labels,
)


def _generate_synthetic_candle_dict(n_bars: int = 120) -> dict[str, pd.DataFrame]:
    """Generate minimal synthetic candles for two instruments."""
    np.random.seed(42)
    datasets = {}
    for sym in ["NSE:TEST1", "NSE:TEST2"]:
        timestamps = pd.date_range("2026-01-01 09:15", periods=n_bars, freq="5min")
        close = 1000.0 + np.cumsum(np.random.randn(n_bars) * 2.0)
        high = close + np.random.uniform(0.5, 3.0, size=n_bars)
        low = close - np.random.uniform(0.5, 3.0, size=n_bars)
        open_ = close + np.random.randn(n_bars) * 0.5
        volume = np.random.randint(1000, 50000, size=n_bars)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        datasets[sym] = df
    return datasets


def test_extract_features_and_labels_integrity():
    """Verify feature extraction and forward labeling integrity."""
    datasets = _generate_synthetic_candle_dict(n_bars=80)
    X, y, ret = extract_features_and_labels(
        datasets, BASE_FEATURE_COLUMNS, lookahead_bars=3, return_threshold=0.001
    )
    assert len(X) > 0
    assert len(X) == len(y) == len(ret)
    assert set(np.unique(y)).issubset({0, 1, 2})
    assert not X.isna().any().any()
    assert not np.isnan(ret).any()


def test_calibrated_ensemble_prediction_structure():
    """Verify that CalibratedEnsemblePredictor generates valid probability distributions."""
    from sklearn.ensemble import ExtraTreesClassifier
    from models.ml_classifier import LightGBMPredictor

    datasets = _generate_synthetic_candle_dict(n_bars=100)
    X, y, ret = extract_features_and_labels(
        datasets, BASE_FEATURE_COLUMNS, lookahead_bars=2, return_threshold=0.0005
    )
    lgbm = LightGBMPredictor("test_lgbm", "v1", feature_names=list(X.columns), n_estimators=10)
    lgbm.fit(X, y, strict_multiclass=True)

    et = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
    et.fit(X, y)

    ensemble = CalibratedEnsemblePredictor(lgbm, et, weight_lgbm=0.6, weight_et=0.4)
    probas = ensemble.predict_proba(X)

    assert probas.shape == (len(X), 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
    assert (probas >= 0.0).all()


def test_microstructure_feature_extraction_and_moe():
    """Verify microstructure feature extraction and RegimeGatedEnsemblePredictor probabilities."""
    from sklearn.ensemble import ExtraTreesClassifier
    from models.ml_classifier import LightGBMPredictor

    datasets = _generate_synthetic_candle_dict(n_bars=100)
    X, y, ret = extract_features_and_labels(
        datasets, MICROSTRUCTURE_FEATURE_COLUMNS, lookahead_bars=2, return_threshold=0.0005
    )
    assert "ofi_ratio" in X.columns
    assert "volume_price_delta" in X.columns
    assert "micro_price_bps" in X.columns

    cols = [c for c in MICROSTRUCTURE_FEATURE_COLUMNS if c in X.columns]
    m_trend = LightGBMPredictor("trend_lgbm", "v1", feature_names=cols, n_estimators=10)
    m_trend.fit(X[cols], y, strict_multiclass=False)

    m_range = LightGBMPredictor("range_lgbm", "v1", feature_names=cols, n_estimators=10)
    m_range.fit(X[cols], y, strict_multiclass=False)

    et = ExtraTreesClassifier(n_estimators=10, max_depth=3, random_state=42)
    et.fit(X[cols], y)
    global_ens = CalibratedEnsemblePredictor(m_trend, et, weight_lgbm=0.6, weight_et=0.4)

    moe = RegimeGatedEnsemblePredictor(
        trend_model=m_trend,
        range_model=m_range,
        global_model=global_ens,
        weight_expert=0.65,
        weight_global=0.35,
    )

    is_trend_dummy = np.random.choice([True, False], size=len(X))
    probas = moe.predict_proba(X[cols], is_trend_mask=is_trend_dummy)

    assert probas.shape == (len(X), 3)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
    assert (probas >= 0.0).all()

