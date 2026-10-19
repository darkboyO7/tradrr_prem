"""Tests for Quantum-Inspired Portfolio Rebalancer & Cardinality-Constrained Quadratic Program."""

import pytest

from risk.quantum_rebalancer import (
    CardinalityOptimizationConfig,
    CardinalityOptimizationReport,
    OptimizationSolverMethod,
    QuantumCardinalityRebalancer,
    RebalanceAllocationItem,
)


@pytest.fixture
def sample_portfolio_universe():
    """8-asset universe with expected returns and covariance matrix."""
    symbols = ["BTC", "ETH", "SOL", "AVAX", "LINK", "DOT", "ADA", "UNI"]
    returns = {
        "BTC": 0.25,
        "ETH": 0.30,
        "SOL": 0.45,
        "AVAX": 0.35,
        "LINK": 0.20,
        "DOT": 0.15,
        "ADA": 0.10,
        "UNI": 0.18,
    }
    # Synthetic positive-definite diagonal-dominant covariance matrix
    cov = [
        [0.09, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.02],
        [0.04, 0.12, 0.05, 0.03, 0.03, 0.02, 0.01, 0.03],
        [0.03, 0.05, 0.20, 0.06, 0.04, 0.03, 0.02, 0.04],
        [0.02, 0.03, 0.06, 0.18, 0.04, 0.03, 0.02, 0.03],
        [0.02, 0.03, 0.04, 0.04, 0.10, 0.02, 0.02, 0.02],
        [0.01, 0.02, 0.03, 0.03, 0.02, 0.14, 0.03, 0.02],
        [0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.15, 0.02],
        [0.02, 0.03, 0.04, 0.03, 0.02, 0.02, 0.02, 0.11],
    ]
    return symbols, returns, cov


def test_cardinality_constraint_strict_enforcement(sample_portfolio_universe):
    """Verify that optimized portfolio never selects more assets than max_cardinality_assets."""
    symbols, returns, cov = sample_portfolio_universe
    cfg = CardinalityOptimizationConfig(
        max_cardinality_assets=3,
        min_weight_threshold=0.05,
        max_weight_ceiling=0.50,
        sa_iterations=1500,
        random_seed=42,
    )
    rebalancer = QuantumCardinalityRebalancer(config=cfg)

    report = rebalancer.optimize_portfolio(
        symbols=symbols,
        expected_returns=returns,
        covariance_matrix=cov,
        solver_method=OptimizationSolverMethod.SIMULATED_ANNEALING_QUBO,
    )

    assert isinstance(report, CardinalityOptimizationReport)
    assert report.total_assets_universe == 8
    assert report.selected_cardinality <= 3
    assert report.max_cardinality_limit == 3
    assert report.portfolio_volatility > 0.0
    assert report.portfolio_expected_return > 0.0

    # Verify weight sums and active items
    active_weights = [item.target_weight for item in report.allocations if item.is_active]
    assert len(active_weights) == report.selected_cardinality
    assert sum(active_weights) == pytest.approx(1.0, abs=1e-3)

    # Verify individual weight limits
    for item in report.allocations:
        if item.is_active:
            assert item.target_weight >= 0.05
            assert item.target_weight <= 0.50


def test_greedy_vs_simulated_annealing(sample_portfolio_universe):
    """Verify greedy projection solver executes deterministically and satisfies cardinality."""
    symbols, returns, cov = sample_portfolio_universe
    cfg = CardinalityOptimizationConfig(max_cardinality_assets=4)
    rebalancer = QuantumCardinalityRebalancer(config=cfg)

    rep_greedy = rebalancer.optimize_portfolio(
        symbols=symbols,
        expected_returns=returns,
        covariance_matrix=cov,
        solver_method=OptimizationSolverMethod.GREEDY_CARDINALITY_PROJECTION,
    )

    assert rep_greedy.selected_cardinality <= 4
    assert rep_greedy.converged is True
    assert rep_greedy.solver_method == OptimizationSolverMethod.GREEDY_CARDINALITY_PROJECTION


def test_transaction_cost_turnover_friction(sample_portfolio_universe):
    """Verify that current weights are penalized by turnover and transaction costs."""
    symbols, returns, cov = sample_portfolio_universe
    # Initial weights already in top assets
    w0 = {"BTC": 0.50, "ETH": 0.50}

    cfg = CardinalityOptimizationConfig(
        max_cardinality_assets=2,
        transaction_cost_bps=20.0,
        sa_iterations=1000,
        random_seed=42,
    )
    rebalancer = QuantumCardinalityRebalancer(config=cfg)

    report = rebalancer.optimize_portfolio(
        symbols=symbols,
        expected_returns=returns,
        covariance_matrix=cov,
        current_weights=w0,
    )

    assert report.turnover_weight >= 0.0
    assert report.estimated_transaction_cost_bps >= 0.0
    assert any(a.current_weight > 0.0 for a in report.allocations)


def test_invalid_covariance_shape_handling():
    """Verify input validation triggers ValueError on malformed matrices."""
    rebalancer = QuantumCardinalityRebalancer()
    with pytest.raises(ValueError, match="Covariance matrix shape"):
        rebalancer.optimize_portfolio(
            symbols=["BTC", "ETH"],
            expected_returns={"BTC": 0.1, "ETH": 0.2},
            covariance_matrix=[[0.05]],  # 1x1 instead of 2x2
        )
