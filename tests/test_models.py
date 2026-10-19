"""Tests for Statistical Baseline, LightGBM Classifier, and Model Registry."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.base import ModelPrediction, SignalAction
from models.ml_classifier import LightGBMPredictor
from models.registry import ModelRegistry
from models.statistical import StatisticalBaselineModel


def test_statistical_baseline_model(tmp_path: Path):
    model = StatisticalBaselineModel()
    df = pd.DataFrame({
        "rsi_14": [25.0, 75.0, 50.0],
        "adx_14": [15.0, 15.0, 30.0],
        "ema_20_slope_5": [0.0, 0.0, 0.05],
        "dist_ema_20_pct": [0.0, 0.0, 0.02],
    })

    probas = model.predict_proba(df)
    assert probas.shape == (3, 3)

    preds = model.predict(df)
    assert len(preds) == 3
    # Row 0: Oversold RSI (25) in low ADX -> Buy
    assert preds[0].action == SignalAction.BUY
    # Row 1: Overbought RSI (75) in low ADX -> Sell
    assert preds[1].action == SignalAction.SELL
    # Row 2: Strong trend with positive slope -> Buy
    assert preds[2].action == SignalAction.BUY

    # Test save and load
    save_dir = model.save(tmp_path / "stat_model")
    loaded = StatisticalBaselineModel.load(save_dir)
    assert loaded.metadata.model_id == model.metadata.model_id


def test_lightgbm_predictor_fit_predict_save(tmp_path: Path):
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "return_1": np.random.normal(0, 0.01, n),
        "log_return_5": np.random.normal(0, 0.02, n),
        "candle_range_pct": np.random.uniform(0.005, 0.02, n),
        "candle_body_ratio": np.random.uniform(0.1, 0.9, n),
        "rsi_14": np.random.uniform(20, 80, n),
        "macd_hist": np.random.normal(0, 0.5, n),
        "adx_14": np.random.uniform(10, 50, n),
        "atr_14": np.random.uniform(1, 5, n),
        "rvol_20": np.random.uniform(0.5, 3.0, n),
        "volatility_regime": np.random.randint(0, 4, n),
    })
    # Target: 0 (Sell), 1 (Neutral), 2 (Buy)
    y = np.random.randint(0, 3, n)

    predictor = LightGBMPredictor(n_estimators=10, max_depth=3)
    predictor.fit(X, y)
    assert predictor.is_fitted

    probas = predictor.predict_proba(X.iloc[:5])
    assert probas.shape == (5, 3)
    # Probabilities sum to 1
    assert np.allclose(probas.sum(axis=1), 1.0)

    preds = predictor.predict(X.iloc[:5])
    assert len(preds) == 5
    assert isinstance(preds[0], ModelPrediction)

    importances = predictor.get_feature_importance()
    assert len(importances) == len(predictor.metadata.feature_names)
    assert pytest.approx(sum(importances.values()), rel=1e-2) == 1.0

    # Serialization with SHA-256 verification
    save_dir = predictor.save(tmp_path / "lgbm_model")
    assert predictor.metadata.sha256_hash is not None

    loaded_pred = LightGBMPredictor.load(save_dir)
    assert loaded_pred.is_fitted
    assert loaded_pred.metadata.sha256_hash == predictor.metadata.sha256_hash


def test_model_registry_save_load(tmp_path: Path):
    registry = ModelRegistry(base_dir=tmp_path)
    model = StatisticalBaselineModel(model_id="test_reg_stat")

    saved_path = registry.save_model(model)
    assert saved_path.exists()

    loaded = registry.load_model("test_reg_stat", "v1.0.0", model_type="StatisticalBaselineModel")
    assert loaded.metadata.model_id == "test_reg_stat"
