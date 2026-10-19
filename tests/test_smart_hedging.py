"""Tests for Cross-Asset Smart Hedging and Real-Time Greeks Rebalancer."""

from datetime import datetime, timezone
import math
import pytest

from analysis.derivatives import DerivativePortfolioReport
from execution.hedging import (
    HedgeInstrument,
    HedgeOrder,
    HedgeTriggerType,
    SmartHedgeConfig,
    SmartHedgePlan,
    SmartHedgingEngine,
)
from execution.simulated import SimulatedExecutionClient


def create_sample_report(net_delta: float, net_gamma: float = 0.002, net_vega: float = 1500.0) -> DerivativePortfolioReport:
    """Helper to generate mock DerivativePortfolioReport."""
    return DerivativePortfolioReport(
        total_market_value_usd=50000.0,
        net_delta=net_delta,
        dollar_delta_usd=net_delta * 65000.0,
        net_gamma=net_gamma,
        dollar_gamma_usd=1500.0,
        net_vega_1pct_usd=net_vega,
        net_theta_daily_usd=-250.0,
        net_rho_1pct_usd=50.0,
        net_vanna_usd=10.0,
        net_volga_usd=5.0,
        delta_neutral_hedge_shares=-net_delta,
        timestamp=datetime.now(timezone.utc),
    )


def test_smart_hedging_deadband_target_aligned():
    """Verify that delta within tolerance band does not trigger unnecessary turnover."""
    engine = SmartHedgingEngine(config=SmartHedgeConfig(delta_tolerance=0.50))
    report = create_sample_report(net_delta=0.35)

    plan = engine.generate_hedge_plan(
        portfolio_greeks=report,
        spot_symbol="BTC-USD",
        spot_price=65000.0,
        perp_symbol="BTC-PERP",
        perp_price=65010.0,
    )

    assert plan.hedging_required is False
    assert plan.trigger_reason == HedgeTriggerType.TARGET_ALIGNED
    assert len(plan.orders) == 0
    assert plan.total_turnover_usd == 0.0
    assert plan.projected_post_hedge_delta == 0.35


def test_smart_hedging_delta_breach_long_delta():
    """Verify that long delta excess triggers sell hedge orders with market impact estimation."""
    engine = SmartHedgingEngine(config=SmartHedgeConfig(delta_tolerance=0.50))
    report = create_sample_report(net_delta=3.20)

    plan = engine.generate_hedge_plan(
        portfolio_greeks=report,
        spot_symbol="BTC-USD",
        spot_price=65000.0,
        perp_symbol="BTC-PERP",
        perp_price=65000.0,
    )

    assert plan.hedging_required is True
    assert plan.trigger_reason == HedgeTriggerType.DELTA_BAND_BREACH
    assert len(plan.orders) == 1

    order = plan.orders[0]
    assert order.side == "SELL"
    assert order.quantity == 3.20
    assert order.symbol == "BTC-PERP"
    assert order.instrument_type == HedgeInstrument.PERPETUAL
    assert order.notional_usd == 3.20 * 65000.0
    assert order.projected_post_hedge_delta == 0.0
    assert order.estimated_cost.total_cost_usd > 0.0
    assert math.isclose(plan.total_estimated_cost_usd, order.estimated_cost.total_cost_usd, abs_tol=0.01)


def test_smart_hedging_delta_breach_short_delta():
    """Verify that short delta excess triggers buy hedge orders."""
    engine = SmartHedgingEngine(config=SmartHedgeConfig(delta_tolerance=0.50))
    report = create_sample_report(net_delta=-2.50)

    plan = engine.generate_hedge_plan(
        portfolio_greeks=report,
        spot_symbol="BTC-USD",
        spot_price=65000.0,
    )

    assert plan.hedging_required is True
    assert len(plan.orders) == 1
    order = plan.orders[0]
    assert order.side == "BUY"
    assert order.quantity == 2.50
    assert order.symbol == "BTC-USD"
    assert order.instrument_type == HedgeInstrument.SPOT


def test_smart_hedging_strategy_selection_twap_vs_immediate():
    """Verify routing via TWAP for orders exceeding notional threshold vs IMMEDIATE for small orders."""
    config = SmartHedgeConfig(delta_tolerance=0.50, twap_threshold_usd=100000.0)
    engine = SmartHedgingEngine(config=config)

    # Large order: 2.0 BTC @ 65,000 = $130,000 > $100,000 -> TWAP
    large_rep = create_sample_report(net_delta=2.0)
    large_plan = engine.generate_hedge_plan(
        portfolio_greeks=large_rep,
        spot_symbol="BTC-USD",
        spot_price=65000.0,
    )
    assert large_plan.orders[0].execution_strategy == "TWAP"

    # Small order: 0.8 BTC @ 65,000 = $52,000 < $100,000 -> IMMEDIATE
    small_rep = create_sample_report(net_delta=0.8)
    small_plan = engine.generate_hedge_plan(
        portfolio_greeks=small_rep,
        spot_symbol="BTC-USD",
        spot_price=65000.0,
    )
    assert small_plan.orders[0].execution_strategy == "IMMEDIATE"


@pytest.mark.asyncio
async def test_smart_hedging_execution_dispatch():
    """Verify dispatching hedge plan orders against execution client."""
    client = SimulatedExecutionClient()
    await client.connect()

    engine = SmartHedgingEngine()
    rep = create_sample_report(net_delta=1.50)
    plan = engine.generate_hedge_plan(
        portfolio_greeks=rep,
        spot_symbol="BTC-USD",
        spot_price=65000.0,
    )

    results = await engine.execute_hedge_plan(plan=plan, client=client)
    assert len(results) == 1
    assert results[0]["status"] == "FILLED"
    assert results[0]["symbol"] == "BTC-USD"
    assert results[0]["filled_quantity"] == 1.50
