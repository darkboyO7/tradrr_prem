"""Tests for Portfolio Rebalancer."""

import pytest

from risk.rebalancing import PortfolioRebalancer, RebalanceOrder, RebalancePlan


def test_rebalancing_plan_generation_and_sequencing():
    """Verify that rebalancer sequences SELLs before BUYs and calculates turnover."""
    rebalancer = PortfolioRebalancer(deadband_drift_pct=0.02)

    # Initial state: 1 BTC ($60,000) + 10 ETH ($30,000) + $10,000 Cash = $100,000 Equity
    # Current weights: BTC 60%, ETH 30%, Cash 10%
    current_holdings = {"BTC-USD": 1.0, "ETH-USD": 10.0}
    current_cash = 10_000.0
    current_prices = {"BTC-USD": 60_000.0, "ETH-USD": 3000.0}

    # Target: BTC 40% (needs to SELL 20%), ETH 50% (needs to BUY 20%), Cash 10%
    target_weights = {"BTC-USD": 0.40, "ETH-USD": 0.50}

    plan = rebalancer.generate_plan(
        current_holdings=current_holdings,
        current_cash=current_cash,
        current_prices=current_prices,
        target_weights=target_weights,
    )

    assert isinstance(plan, RebalancePlan)
    assert plan.total_equity == 100_000.0
    assert len(plan.orders) == 2

    # SELL order must come FIRST
    assert plan.orders[0].side == "SELL"
    assert plan.orders[0].symbol == "BTC-USD"
    # Target value = 40,000 -> target units = 40,000 / 60,000 = 0.666667 BTC -> sell ~0.333333 BTC
    assert plan.orders[0].order_quantity == pytest.approx(0.333333, rel=1e-4)

    # BUY order must come SECOND
    assert plan.orders[1].side == "BUY"
    assert plan.orders[1].symbol == "ETH-USD"
    # Target value = 50,000 -> target units = 50,000 / 3,000 = 16.666667 ETH -> buy ~6.666667 ETH
    assert plan.orders[1].order_quantity == pytest.approx(6.666667, rel=1e-4)

    assert plan.estimated_turnover_usd == pytest.approx(40_000.0, rel=0.01)
    assert plan.estimated_total_cost_usd > 0
    assert plan.estimated_cost_bps > 0


def test_rebalancing_deadband_filters_minor_drift():
    """Verify that assets with drift within deadband threshold are ignored."""
    # Deadband = 3% drift tolerance
    rebalancer = PortfolioRebalancer(deadband_drift_pct=0.03)

    # Current: BTC $50,000 (50%), Cash $50,000 (50%) -> Total $100,000
    current_holdings = {"BTC-USD": 1.0}
    current_cash = 50_000.0
    current_prices = {"BTC-USD": 50_000.0}

    # Target: BTC 51% (drift is only 1%, well below 3% threshold)
    target_weights = {"BTC-USD": 0.51}

    plan = rebalancer.generate_plan(
        current_holdings=current_holdings,
        current_cash=current_cash,
        current_prices=current_prices,
        target_weights=target_weights,
    )

    # No orders should be generated because drift < deadband
    assert len(plan.orders) == 0
    assert plan.estimated_turnover_usd == 0.0
