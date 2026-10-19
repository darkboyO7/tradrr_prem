"""Tests for Quantitative Systematic Factor Attribution Model."""

import numpy as np
import pandas as pd
import pytest

from risk.factor_model import (
    AssetFactorExposure,
    PortfolioFactorAttribution,
    SystematicFactorModel,
)


@pytest.fixture
def multi_asset_price_data() -> pd.DataFrame:
    """Generate 60 days of multi-asset prices and volumes."""
    rng = np.random.default_rng(101)
    dates = pd.date_range("2024-01-01", periods=60, freq="1D")

    # Generate returns
    r_mkt = rng.normal(0.001, 0.025, 60)
    r_btc = r_mkt + rng.normal(0.0, 0.005, 60)
    r_eth = 1.2 * r_mkt + rng.normal(0.0, 0.015, 60)
    r_sol = 1.8 * r_mkt + rng.normal(0.0, 0.025, 60)

    p_btc = 60000.0 * np.exp(np.cumsum(r_btc))
    p_eth = 3000.0 * np.exp(np.cumsum(r_eth))
    p_sol = 120.0 * np.exp(np.cumsum(r_sol))

    return pd.DataFrame({"BTC-USD": p_btc, "ETH-USD": p_eth, "SOL-USD": p_sol}, index=dates)


def test_synthetic_factor_construction(multi_asset_price_data: pd.DataFrame):
    """Verify construction of canonical factor series."""
    model = SystematicFactorModel(benchmark_symbol="BTC-USD")
    factors = model.construct_synthetic_factors(multi_asset_price_data)

    assert set(factors.columns) == {"Market", "Momentum", "Volatility", "Liquidity"}
    assert len(factors) == len(multi_asset_price_data) - 1
    # Check that market factor matches BTC return
    ret_btc = np.log(multi_asset_price_data["BTC-USD"] / multi_asset_price_data["BTC-USD"].shift(1)).dropna()
    np.testing.assert_allclose(factors["Market"].values, ret_btc.values)


def test_asset_factor_analysis(multi_asset_price_data: pd.DataFrame):
    """Verify OLS factor loading and specific risk estimation on an individual asset."""
    model = SystematicFactorModel(benchmark_symbol="BTC-USD")
    factors = model.construct_synthetic_factors(multi_asset_price_data)
    returns = np.log(multi_asset_price_data / multi_asset_price_data.shift(1)).dropna()

    exp = model.analyze_asset(returns["ETH-USD"], factors, symbol="ETH-USD")
    assert isinstance(exp, AssetFactorExposure)
    assert exp.symbol == "ETH-USD"
    assert 0.0 <= exp.r_squared <= 1.0
    assert exp.specific_risk_annualized > 0
    assert "Market" in exp.factor_betas
    # ETH market beta should be around 1.2 based on fixture generation
    assert exp.factor_betas["Market"] > 0.5


def test_portfolio_factor_attribution(multi_asset_price_data: pd.DataFrame):
    """Verify aggregate portfolio factor decomposition."""
    model = SystematicFactorModel(benchmark_symbol="BTC-USD")
    weights = {"BTC-USD": 0.5, "ETH-USD": 0.3, "SOL-USD": 0.2}

    attribution = model.attribute_portfolio(multi_asset_price_data, weights)
    assert isinstance(attribution, PortfolioFactorAttribution)
    assert attribution.portfolio_beta_market > 0.8
    assert 0.0 <= attribution.portfolio_r_squared <= 1.0

    # Risk split must sum to 100%
    assert attribution.systematic_risk_pct + attribution.idiosyncratic_risk_pct == pytest.approx(100.0, rel=1e-3)
    assert len(attribution.asset_exposures) == 3
