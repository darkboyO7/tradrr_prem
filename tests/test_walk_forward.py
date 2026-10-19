"""Tests for Walk-Forward Validation Engine."""

import numpy as np
import pandas as pd

from backtesting.walk_forward import WalkForwardValidator


def test_walk_forward_validation():
    np.random.seed(42)
    bars = 300
    returns = np.random.normal(0.0005, 0.007, bars)
    closes = 60000.0 * np.exp(np.cumsum(returns))
    highs = closes * 1.004
    lows = closes * 0.996
    opens = closes * 0.999
    volumes = np.random.uniform(50, 150, bars)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    validator = WalkForwardValidator(n_splits=3, train_ratio=0.60, embargo_bars=3)
    result = validator.run_walk_forward(df, symbol="BTC-USD", timeframe="1h")

    assert result["n_folds"] > 0
    assert len(result["folds"]) == result["n_folds"]
    assert "oos_equity_curve" in result
    assert "oos_metrics" in result

    metrics = result["oos_metrics"]
    assert isinstance(metrics.sharpe_ratio, float)
    assert isinstance(metrics.max_drawdown_pct, float)
