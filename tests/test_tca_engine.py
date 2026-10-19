"""Tests for Post-Trade Transaction Cost Analysis (TCA) & Slippage Attribution."""

from datetime import datetime, timedelta, timezone
import math
import pytest

from execution.tca import (
    ImplementationShortfallDecomposition,
    OrderBenchmarkType,
    TCAPortfolioReport,
    TCAStatisticalDistribution,
    TradeExecutionRecord,
    TransactionCostAnalysisEngine,
    VenueExecutionScorecard,
)


def create_sample_trade(
    symbol: str = "BTC-USD",
    side: str = "BUY",
    venue: str = "binance",
    order_qty: float = 1.0,
    filled_qty: float = 1.0,
    decision_px: float = 65000.0,
    arrival_px: float = 65002.0,
    arrival_bid: float = 65001.0,
    arrival_ask: float = 65003.0,
    exec_px: float = 65005.0,
    terminal_px: float = 65010.0,
    fee_usd: float = 10.0,
    latency_sec: float = 0.05,
) -> TradeExecutionRecord:
    """Helper to generate mock trade execution record."""
    t0 = datetime.now(timezone.utc)
    t1 = t0 + timedelta(milliseconds=20)
    t2 = t0 + timedelta(seconds=latency_sec)
    return TradeExecutionRecord(
        order_id=f"ord_test_{uuid_sample()}",
        symbol=symbol,
        side=side,
        venue=venue,
        order_quantity=order_qty,
        filled_quantity=filled_qty,
        decision_price=decision_px,
        arrival_price=arrival_px,
        arrival_bid=arrival_bid,
        arrival_ask=arrival_ask,
        execution_price=exec_px,
        terminal_price=terminal_px,
        decision_time=t0,
        arrival_time=t1,
        execution_time=t2,
        venue_fees_usd=fee_usd,
    )


def uuid_sample() -> str:
    import uuid
    return uuid.uuid4().hex[:6]


def test_tca_implementation_shortfall_decomposition():
    """Verify exact mathematical decomposition of Implementation Shortfall."""
    engine = TransactionCostAnalysisEngine()
    trade = create_sample_trade(
        side="BUY",
        order_qty=2.0,
        filled_qty=2.0,
        decision_px=65000.0,
        arrival_px=65002.0,
        arrival_bid=65001.0,
        arrival_ask=65003.0,  # mid = 65002.0
        exec_px=65006.0,
        fee_usd=15.0,
    )

    decomp = engine.decompose_trade(trade)

    # 1. Delay Cost: (65002 - 65000) * 2 = $4.00
    assert math.isclose(decomp.delay_cost_usd, 4.0, abs_tol=0.01)

    # 2. Spread Cost: (65006 - 65002) * 2 = $8.00 (half-spread is $1.00, so effective spread cost is $8.00)
    assert decomp.spread_cost_usd > 0.0

    # 3. Fee drag: $15.00
    assert decomp.fee_drag_usd == 15.0

    # 4. Total Shortfall: (65006 - 65000) * 2 + 15 = 12 + 15 = $27.00
    assert math.isclose(decomp.total_shortfall_usd, 27.0, abs_tol=0.01)
    assert decomp.notional_usd == 2.0 * 65006.0
    assert decomp.total_shortfall_bps > 0.0


def test_tca_price_improvement_detection():
    """Verify detection of execution inside the quoted bid-ask spread."""
    engine = TransactionCostAnalysisEngine()

    # Buy order executed below arrival ask ($65008 vs quoted ask $65010)
    pi_trade = create_sample_trade(
        side="BUY",
        arrival_bid=65000.0,
        arrival_ask=65010.0,
        exec_px=65008.0,
        filled_qty=1.0,
    )
    res_pi = engine.decompose_trade(pi_trade)
    assert res_pi.is_price_improved is True
    assert math.isclose(res_pi.price_improvement_usd, 2.0, abs_tol=0.01)

    # Buy order executed at or above arrival ask -> no price improvement
    no_pi_trade = create_sample_trade(
        side="BUY",
        arrival_bid=65000.0,
        arrival_ask=65010.0,
        exec_px=65012.0,
        filled_qty=1.0,
    )
    res_no_pi = engine.decompose_trade(no_pi_trade)
    assert res_no_pi.is_price_improved is False
    assert res_no_pi.price_improvement_usd == 0.0


def test_tca_opportunity_cost_partial_fill():
    """Verify opportunity cost on unexecuted balance when market moves away."""
    engine = TransactionCostAnalysisEngine()

    # Ordered 2.0 BTC, only filled 1.0 BTC. Market rallied to 65050 terminal
    partial_trade = create_sample_trade(
        side="BUY",
        order_qty=2.0,
        filled_qty=1.0,
        decision_px=65000.0,
        exec_px=65005.0,
        terminal_px=65050.0,
    )
    decomp = engine.decompose_trade(partial_trade)

    # Opportunity cost: (65050 - 65000) * (2.0 - 1.0) = $50.00
    assert math.isclose(decomp.opportunity_cost_usd, 50.0, abs_tol=0.01)
    assert decomp.opportunity_cost_bps > 0.0


def test_tca_venue_scorecards():
    """Verify venue execution quality scorecards and EFQ ratios."""
    engine = TransactionCostAnalysisEngine()

    t_binance = [
        create_sample_trade(venue="binance", arrival_bid=65000.0, arrival_ask=65010.0, exec_px=65004.0),
        create_sample_trade(venue="binance", arrival_bid=65000.0, arrival_ask=65010.0, exec_px=65006.0),
    ]
    t_kraken = [
        create_sample_trade(venue="kraken", arrival_bid=65000.0, arrival_ask=65010.0, exec_px=65012.0),
    ]

    cards = engine.calculate_venue_scorecards(t_binance + t_kraken)
    assert "binance" in cards
    assert "kraken" in cards

    binance_card = cards["binance"]
    assert binance_card.total_orders == 2
    assert binance_card.fill_rate_pct == 100.0
    assert binance_card.effective_to_quoted_ratio < 1.0  # Fills near midpoint have EFQ < 1.0
    assert binance_card.price_improvement_pct == 100.0   # Both executed below ask 65010

    kraken_card = cards["kraken"]
    assert kraken_card.effective_to_quoted_ratio > 1.0   # Executed outside spread


def test_tca_portfolio_report_and_distribution():
    """Verify complete portfolio TCA report aggregation and slippage percentiles."""
    engine = TransactionCostAnalysisEngine()
    trades = [
        create_sample_trade(exec_px=65002.0),
        create_sample_trade(exec_px=65005.0),
        create_sample_trade(exec_px=65008.0),
        create_sample_trade(exec_px=65015.0),
    ]

    report = engine.generate_portfolio_tca_report(trades)
    assert isinstance(report, TCAPortfolioReport)
    assert report.total_orders_analyzed == 4
    assert report.total_notional_usd > 200000.0
    assert report.aggregate_shortfall_usd > 0.0

    dist = report.slippage_distribution
    assert isinstance(dist, TCAStatisticalDistribution)
    assert dist.count == 4
    assert dist.p50_shortfall_bps <= dist.p95_shortfall_bps
    assert dist.p95_shortfall_bps <= dist.p99_shortfall_bps
