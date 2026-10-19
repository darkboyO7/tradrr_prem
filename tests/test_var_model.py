"""Tests for Multi-Asset Covariance and Value-at-Risk (VaR) Engine."""

import numpy as np
import pandas as pd
import pytest

from risk.var_model import PortfolioRiskModel, VaRMetrics


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Generate 100 days of correlated price history for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")

    # Generate correlated return paths
    ret_btc = rng.normal(0.001, 0.03, 100)
    ret_eth = 0.8 * ret_btc + rng.normal(0.0005, 0.02, 100)

    p_btc = 60000.0 * np.exp(np.cumsum(ret_btc))
    p_eth = 3000.0 * np.exp(np.cumsum(ret_eth))

    return pd.DataFrame({"BTC-USD": p_btc, "ETH-USD": p_eth}, index=dates)


def test_compute_returns_and_covariance(sample_price_data: pd.DataFrame):
    """Verify log return calculation and exponentially weighted covariance matrix."""
    model = PortfolioRiskModel(decay_factor=0.94)
    returns = model.compute_returns(sample_price_data)

    assert len(returns) == 99
    assert "BTC-USD" in returns.columns
    assert "ETH-USD" in returns.columns

    cov = model.compute_covariance_matrix(returns, exponential_weighting=True)
    assert cov.shape == (2, 2)
    # Variances must be strictly positive
    assert cov.loc["BTC-USD", "BTC-USD"] > 0
    assert cov.loc["ETH-USD", "ETH-USD"] > 0
    # Positive correlation between BTC and ETH
    assert cov.loc["BTC-USD", "ETH-USD"] > 0


def test_calculate_var_all_methodologies(sample_price_data: pd.DataFrame):
    """Verify calculation of Parametric, Historical, and Monte Carlo VaR & CVaR."""
    model = PortfolioRiskModel()
    weights = {"BTC-USD": 0.6, "ETH-USD": 0.4}

    metrics = model.calculate_var(
        prices_df=sample_price_data,
        weights=weights,
        portfolio_value=100_000.0,
        confidence_level=0.95,
        horizon_days=1,
        monte_carlo_sims=5000,
        random_seed=42,
    )

    assert isinstance(metrics, VaRMetrics)
    assert metrics.portfolio_value == 100_000.0
    assert metrics.confidence_level == 0.95
    assert metrics.portfolio_volatility_daily > 0
    assert metrics.portfolio_volatility_annualized > metrics.portfolio_volatility_daily

    # VaR must be positive numbers
    assert metrics.parametric_var > 0
    assert metrics.historical_var > 0
    assert metrics.monte_carlo_var > 0

    # CVaR (Expected Shortfall) must be >= VaR
    assert metrics.parametric_cvar >= metrics.parametric_var
    assert metrics.historical_cvar >= metrics.historical_var
    assert metrics.monte_carlo_cvar >= metrics.monte_carlo_var


def test_component_var_euler_additivity(sample_price_data: pd.DataFrame):
    """Verify Euler decomposition: sum of Component VaRs equals total Parametric VaR."""
    model = PortfolioRiskModel()
    weights = {"BTC-USD": 0.7, "ETH-USD": 0.3}

    metrics = model.calculate_var(
        prices_df=sample_price_data,
        weights=weights,
        portfolio_value=100_000.0,
        confidence_level=0.95,
    )

    sum_component_var = sum(metrics.component_var.values())
    assert sum_component_var == pytest.approx(metrics.parametric_var, rel=0.01)

    sum_percentage_contrib = sum(metrics.percentage_risk_contribution.values())
    assert sum_percentage_contrib == pytest.approx(100.0, rel=0.01)


def test_var_horizon_scaling(sample_price_data: pd.DataFrame):
    """Verify square-root of time scaling for multi-day VaR horizons."""
    model = PortfolioRiskModel()
    weights = {"BTC-USD": 0.5, "ETH-USD": 0.5}

    metrics_1d = model.calculate_var(sample_price_data, weights, horizon_days=1)
    metrics_4d = model.calculate_var(sample_price_data, weights, horizon_days=4)

    # 4-day VaR should be approximately 2x (sqrt(4)) the 1-day VaR
    assert metrics_4d.parametric_var == pytest.approx(metrics_1d.parametric_var * 2.0, rel=0.05)
