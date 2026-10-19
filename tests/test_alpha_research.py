"""Tests for Quantitative Alpha Research & Multi-Factor Neutralization Pipeline."""

import math
import numpy as np
import pandas as pd
import pytest

from analysis.alpha_research import (
    AlphaFactorType,
    AlphaResearchPipeline,
    AlphaResearchReport,
    FactorEvaluationMetrics,
    NeutralizedFactorScore,
)


def generate_test_price_panel(num_assets: int = 6, num_periods: int = 60) -> pd.DataFrame:
    """Generate deterministic synthetic price panel for factor testing."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=num_periods, freq="D")
    symbols = [f"CRYPTO_{i}" for i in range(num_assets)]

    # Generate correlated random walk
    drift = np.linspace(-0.001, 0.002, num_assets)
    returns = np.random.normal(loc=drift, scale=0.03, size=(num_periods, num_assets))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def test_winsorization_and_standardization():
    """Verify outlier clipping and cross-sectional Z-score standardization."""
    pipeline = AlphaResearchPipeline(winsorize_std=2.5)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])  # Extreme outlier 100.0

    standardized = pipeline.winsorize_and_standardize(s)
    assert len(standardized) == 5
    assert math.isclose(standardized.mean(), 0.0, abs_tol=1e-5)
    assert math.isclose(standardized.std(), 1.0, abs_tol=1e-5)
    assert standardized.iloc[-1] < 3.0  # Outlier properly clipped


def test_factor_orthogonal_neutralization():
    """Verify regression residualization strips out correlation with risk exposures."""
    pipeline = AlphaResearchPipeline()
    np.random.seed(123)
    n = 20
    symbols = [f"ASSET_{i}" for i in range(n)]

    # Risk exposure: Market Beta
    beta = np.random.uniform(0.5, 1.8, n)
    risk_df = pd.DataFrame({"MarketBeta": beta}, index=symbols)

    # Raw factor with high intentional correlation to MarketBeta
    raw_factor = pd.Series(2.5 * beta + np.random.normal(0, 0.3, n), index=symbols)
    raw_corr = raw_factor.corr(risk_df["MarketBeta"])
    assert raw_corr > 0.80  # Confirmed highly correlated

    # Neutralize
    neutralized = pipeline.neutralize_factor(raw_factor, risk_df)
    neutral_corr = neutralized.corr(risk_df["MarketBeta"])

    # Correlation with MarketBeta must be stripped to zero (< 1e-4)
    assert abs(neutral_corr) < 1e-4


def test_alpha_factor_construction():
    """Verify construction of momentum, reversal, and idiosyncratic volatility panels."""
    pipeline = AlphaResearchPipeline()
    prices = generate_test_price_panel(num_assets=5, num_periods=50)

    factors = pipeline.construct_alpha_factors(prices)
    assert AlphaFactorType.MOMENTUM_14D.value in factors
    assert AlphaFactorType.MEAN_REVERSION_3D.value in factors
    assert AlphaFactorType.IDIOSYNCRATIC_VOLATILITY.value in factors

    mom = factors[AlphaFactorType.MOMENTUM_14D.value]
    assert mom.shape == prices.shape
    # 14-period momentum has NaNs only in first 14 rows
    assert not mom.iloc[15:].isna().any().any()


def test_factor_performance_evaluation():
    """Verify Information Coefficient (IC) and quantile monotonicity calculations."""
    pipeline = AlphaResearchPipeline(num_quantiles=3)
    prices = generate_test_price_panel(num_assets=8, num_periods=50)

    factors = pipeline.construct_alpha_factors(prices)
    mom_panel = factors[AlphaFactorType.MOMENTUM_14D.value]
    fwd_ret = prices.pct_change(1).shift(-1)

    metrics = pipeline.evaluate_factor_performance(mom_panel, fwd_ret, "MOM_14D")
    assert isinstance(metrics, FactorEvaluationMetrics)
    assert metrics.factor_name == "MOM_14D"
    assert "Q1" in metrics.quantile_returns
    assert "Q3" in metrics.quantile_returns
    assert abs(metrics.ic_mean) <= 1.0
    assert abs(metrics.rank_ic_mean) <= 1.0
    assert -1.0 <= metrics.quantile_monotonicity_score <= 1.0


def test_alpha_pipeline_end_to_end():
    """Verify end-to-end alpha research report synthesis and ranking."""
    pipeline = AlphaResearchPipeline(num_quantiles=3)
    prices = generate_test_price_panel(num_assets=6, num_periods=45)

    report = pipeline.run_alpha_pipeline(prices, forward_horizon_periods=1)
    assert isinstance(report, AlphaResearchReport)
    assert report.num_assets == 6
    assert report.num_periods == 45
    assert len(report.factor_evaluations) >= 3
    assert len(report.top_ranked_assets) <= 3
    assert len(report.asset_scores) == 6

    # Verify score structure for individual asset
    first_sym = prices.columns[0]
    score = report.asset_scores[first_sym]
    assert isinstance(score, NeutralizedFactorScore)
    assert 0.0 <= score.percentile_rank <= 1.0
