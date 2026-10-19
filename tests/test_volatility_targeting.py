"""Tests for Dynamic Volatility Targeting Engine."""

import numpy as np
import pandas as pd
import pytest

from risk.volatility_targeting import (
    VolatilityTargetConfig,
    VolatilityTargetResult,
    VolatilityTargetingEngine,
)


@pytest.fixture
def high_vol_price_data() -> pd.DataFrame:
    """Generate price series with high volatility (~40% annualized)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="1D")
    ret_high = rng.normal(0.0, 0.04, 60)  # ~76% annualized
    p = 50000.0 * np.exp(np.cumsum(ret_high))
    return pd.DataFrame({"BTC-USD": p}, index=dates)


@pytest.fixture
def low_vol_price_data() -> pd.DataFrame:
    """Generate price series with very low volatility (~8% annualized)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="1D")
    ret_low = rng.normal(0.0, 0.004, 60)  # ~7.6% annualized
    p = 50000.0 * np.exp(np.cumsum(ret_low))
    return pd.DataFrame({"BTC-USD": p}, index=dates)


def test_volatility_targeting_deleveraging_high_vol(high_vol_price_data: pd.DataFrame):
    """Verify that high volatility forces portfolio deleveraging (multiplier < 1.0)."""
    config = VolatilityTargetConfig(target_volatility_annualized=0.15, min_leverage=0.1, max_leverage=1.5)
    engine = VolatilityTargetingEngine(config=config)

    base_weights = {"BTC-USD": 1.0}
    res = engine.calculate_target_weights(high_vol_price_data, base_weights)

    assert isinstance(res, VolatilityTargetResult)
    assert res.realized_volatility_annualized > config.target_volatility_annualized
    # Multiplier must de-lever below 1.0
    assert res.applied_multiplier < 1.0
    assert res.target_weights["BTC-USD"] < 1.0
    assert res.cash_buffer_pct > 0.0


def test_volatility_targeting_leveraging_low_vol(low_vol_price_data: pd.DataFrame):
    """Verify that low volatility expands portfolio exposure (multiplier > 1.0 up to max_leverage)."""
    config = VolatilityTargetConfig(target_volatility_annualized=0.20, min_leverage=0.1, max_leverage=1.5)
    engine = VolatilityTargetingEngine(config=config)

    base_weights = {"BTC-USD": 0.8}
    res = engine.calculate_target_weights(low_vol_price_data, base_weights)

    assert res.realized_volatility_annualized < config.target_volatility_annualized
    # Multiplier should expand allocation up to max_leverage
    assert res.applied_multiplier > 1.0
    assert res.applied_multiplier <= config.max_leverage
    assert res.target_weights["BTC-USD"] > 0.8


def test_volatility_targeting_multi_asset():
    """Verify multi-asset covariance and weight scaling."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=50, freq="1D")
    p_btc = 60000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 50)))
    p_eth = 3000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.025, 50)))
    df = pd.DataFrame({"BTC-USD": p_btc, "ETH-USD": p_eth}, index=dates)

    engine = VolatilityTargetingEngine(VolatilityTargetConfig(target_volatility_annualized=0.15))
    base_weights = {"BTC-USD": 0.6, "ETH-USD": 0.4}
    res = engine.calculate_target_weights(df, base_weights)

    assert "BTC-USD" in res.target_weights
    assert "ETH-USD" in res.target_weights
    # Ratio between weights should remain proportional
    ratio_base = base_weights["BTC-USD"] / base_weights["ETH-USD"]
    ratio_target = res.target_weights["BTC-USD"] / res.target_weights["ETH-USD"]
    assert ratio_target == pytest.approx(ratio_base, rel=1e-3)
