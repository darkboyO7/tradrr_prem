"""Tests for Institutional Dynamic Risk Factor Covariance Shrinkage & Ledoit-Wolf Estimator."""

import numpy as np
import pytest

from risk.covariance_shrinkage import (
    CovarianceShrinkageEngine,
    MinimumVarianceWeights,
    ShrinkageResult,
    ShrinkageTargetType,
)


@pytest.fixture
def sample_returns():
    """Generate reproducible synthetic multi-asset daily returns (T=100, N=5)."""
    np.random.seed(42)
    T, N = 100, 5
    # True factor structure with correlated latent factors
    factors = np.random.randn(T, 2)
    loadings = np.array([
        [0.8, 0.1],
        [0.7, 0.3],
        [0.2, 0.9],
        [0.1, 0.8],
        [0.5, 0.5],
    ]).T  # shape (2, 5)
    idiosyncratic = np.random.randn(T, N) * 0.2
    returns = factors @ loadings + idiosyncratic
    return returns, ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def test_constant_correlation_shrinkage(sample_returns):
    """Verify Ledoit-Wolf constant correlation shrinkage estimation."""
    returns, symbols = sample_returns
    engine = CovarianceShrinkageEngine(
        default_target=ShrinkageTargetType.CONSTANT_CORRELATION,
        annualization_factor=252.0,
    )
    result = engine.estimate_covariance(returns, symbols=symbols)

    assert isinstance(result, ShrinkageResult)
    assert result.symbols == symbols
    assert result.asset_count_n == 5
    assert result.sample_size_t == 100
    assert 0.0 <= result.shrinkage_intensity <= 1.0
    assert result.is_positive_definite is True

    # Shrunk condition number should typically improve or remain finite
    assert result.condition_number_shrunk < float("inf")
    assert result.condition_number_shrunk <= result.condition_number_sample * 1.5

    # Shrunk correlation diagonal should be exactly 1.0
    corr = result.shrunk_correlation
    assert np.allclose(np.diag(corr), 1.0)
    assert np.all(corr >= -1.0 - 1e-6)
    assert np.all(corr <= 1.0 + 1e-6)

    # Shrunk covariance must be symmetric
    cov = result.shrunk_covariance
    assert np.allclose(cov, cov.T)


def test_diagonal_variance_and_scaled_identity_targets(sample_returns):
    """Test diagonal and scaled identity shrinkage targets."""
    returns, symbols = sample_returns
    engine = CovarianceShrinkageEngine()

    res_diag = engine.estimate_covariance(
        returns, symbols=symbols, target_type=ShrinkageTargetType.DIAGONAL_VARIANCE
    )
    assert res_diag.target_type == ShrinkageTargetType.DIAGONAL_VARIANCE
    assert 0.0 <= res_diag.shrinkage_intensity <= 1.0
    assert res_diag.is_positive_definite is True

    res_ident = engine.estimate_covariance(
        returns, symbols=symbols, target_type=ShrinkageTargetType.SCALED_IDENTITY
    )
    assert res_ident.target_type == ShrinkageTargetType.SCALED_IDENTITY
    assert 0.0 <= res_ident.shrinkage_intensity <= 1.0
    assert res_ident.is_positive_definite is True


def test_ill_conditioned_fat_matrix():
    """Verify behavior when N > T (ill-conditioned / rank-deficient sample)."""
    np.random.seed(123)
    T, N = 10, 25  # rank at most T-1 = 9, while N = 25
    returns = np.random.randn(T, N) * 0.01

    engine = CovarianceShrinkageEngine()
    result = engine.estimate_covariance(returns)

    # Sample covariance is strictly singular or severely ill-conditioned
    assert result.condition_number_sample > 1e6

    # Shrunk covariance regularizes the spectrum
    assert result.condition_number_shrunk < result.condition_number_sample
    assert result.condition_number_shrunk < 1e5
    assert result.is_positive_definite is True
    assert result.effective_rank >= 1.0


def test_minimum_variance_portfolio(sample_returns):
    """Test analytical unconstrained and long-only minimum-variance allocations."""
    returns, symbols = sample_returns
    engine = CovarianceShrinkageEngine()
    result = engine.estimate_covariance(returns, symbols=symbols)

    # Long-only allocation
    mv_long = engine.compute_minimum_variance_portfolio(result, long_only=True)
    assert isinstance(mv_long, MinimumVarianceWeights)
    assert len(mv_long.weights) == 5
    assert np.isclose(sum(mv_long.weights.values()), 1.0)
    for w in mv_long.weights.values():
        assert w >= -1e-6

    assert mv_long.portfolio_volatility > 0.0
    assert mv_long.diversification_ratio >= 1.0  # By Cauchy-Schwarz / sub-additivity
    assert 1.0 <= mv_long.effective_constituents <= 5.0

    # Unconstrained allocation
    mv_unconstrained = engine.compute_minimum_variance_portfolio(result, long_only=False)
    assert np.isclose(sum(mv_unconstrained.weights.values()), 1.0)
    # Variance of unconstrained minimum variance should be <= long-only
    assert mv_unconstrained.portfolio_variance <= mv_long.portfolio_variance + 1e-6


def test_custom_shrinkage_intensity(sample_returns):
    """Test custom user-specified shrinkage delta overrides."""
    returns, symbols = sample_returns
    engine = CovarianceShrinkageEngine()

    # Zero shrinkage = pure sample covariance
    res_zero = engine.estimate_covariance(
        returns, symbols=symbols, custom_shrinkage_delta=0.0
    )
    assert res_zero.shrinkage_intensity == 0.0
    assert np.allclose(res_zero.shrunk_covariance, res_zero.sample_covariance)

    # Full shrinkage = pure target covariance
    res_full = engine.estimate_covariance(
        returns, symbols=symbols, custom_shrinkage_delta=1.0
    )
    assert res_full.shrinkage_intensity == 1.0
    assert np.allclose(res_full.shrunk_covariance, res_full.target_covariance)


def test_edge_cases_and_serialization():
    """Test single-asset handling, input validation, and dictionary serialization."""
    engine = CovarianceShrinkageEngine()

    # Single asset
    single_res = engine.estimate_covariance([[0.01], [0.02], [-0.01], [0.005]], symbols=["BTC"])
    assert single_res.asset_count_n == 1
    assert single_res.shrinkage_intensity == 0.0

    mv = engine.compute_minimum_variance_portfolio(single_res)
    assert mv.weights == {"BTC": 1.0}

    # Serialization
    d = single_res.to_dict()
    assert d["symbols"] == ["BTC"]
    assert "shrunk_covariance" in d
    assert mv.to_dict()["effective_constituents"] == 1.0

    # Validation errors
    with pytest.raises(ValueError, match="must be 2-dimensional"):
        engine.estimate_covariance([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="at least 3"):
        engine.estimate_covariance([[0.01, 0.02], [0.03, 0.04]])

    with pytest.raises(ValueError, match="match column count"):
        engine.estimate_covariance(
            [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]],
            symbols=["A"],
        )
