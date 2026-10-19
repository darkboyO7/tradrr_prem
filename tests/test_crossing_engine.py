"""Tests for Institutional Internal Matching & Crossing Engine."""

import math
import pytest

from execution.crossing import (
    CrossingFill,
    CrossingPoolMetrics,
    CrossingPriority,
    CrossingResult,
    CrossingSide,
    CrossingTimeInForce,
    InternalCrossingEngine,
    InternalCrossingOrder,
)


def test_crossing_engine_exact_match_and_savings():
    """Verify midpoint execution, full spread savings, and taker fee elimination."""
    engine = InternalCrossingEngine(default_taker_fee_rate=0.0005)

    # 1. Post resting SELL order of 1.0 BTC
    sell_order = InternalCrossingOrder(
        symbol="BTC-USD",
        side=CrossingSide.SELL,
        quantity=1.0,
        remaining_quantity=1.0,
        limit_price=65000.0,
        time_in_force=CrossingTimeInForce.RESTING,
    )
    res_sell = engine.cross_order(sell_order, current_bid=65000.0, current_ask=65010.0)
    assert res_sell.filled_quantity == 0.0
    assert res_sell.remaining_unfilled_quantity == 1.0

    # 2. Incoming BUY order of 1.0 BTC (IOC)
    buy_order = InternalCrossingOrder(
        symbol="BTC-USD",
        side=CrossingSide.BUY,
        quantity=1.0,
        remaining_quantity=1.0,
        time_in_force=CrossingTimeInForce.IOC,
    )
    res_buy = engine.cross_order(buy_order, current_bid=65000.0, current_ask=65010.0)

    # Expected midpoint: 65,005.0
    assert res_buy.filled_quantity == 1.0
    assert res_buy.remaining_unfilled_quantity == 0.0
    assert res_buy.average_match_price == 65005.0
    assert len(res_buy.fills) == 1

    # Full spread savings: (65010 - 65000) * 1.0 = $10.00
    assert math.isclose(res_buy.total_spread_savings_usd, 10.0, abs_tol=0.01)
    # Fee savings: 2 * 0.0005 * 65005 = $65.01
    assert math.isclose(res_buy.total_fee_savings_usd, 65.005, abs_tol=0.05)


def test_crossing_engine_partial_fill_and_resting_remainder():
    """Verify partial execution against resting liquidity and posting remainder."""
    engine = InternalCrossingEngine()

    # Resting SELL: 0.5 BTC
    engine.cross_order(
        InternalCrossingOrder(
            symbol="ETH-USD",
            side=CrossingSide.SELL,
            quantity=0.5,
            remaining_quantity=0.5,
            time_in_force=CrossingTimeInForce.RESTING,
        ),
        current_bid=3500.0,
        current_ask=3502.0,
    )

    # Incoming BUY: 1.5 ETH (RESTING remainder)
    buy_order = InternalCrossingOrder(
        symbol="ETH-USD",
        side=CrossingSide.BUY,
        quantity=1.5,
        remaining_quantity=1.5,
        time_in_force=CrossingTimeInForce.RESTING,
    )
    res = engine.cross_order(buy_order, current_bid=3500.0, current_ask=3502.0)

    assert res.filled_quantity == 0.5
    assert res.remaining_unfilled_quantity == 1.0
    assert res.average_match_price == 3501.0

    # Book should now have 0 sells and 1 buy with 1.0 ETH
    book = engine.get_resting_book("ETH-USD")
    assert len(book["resting_sells"]) == 0
    assert len(book["resting_buys"]) == 1
    assert book["resting_buys"][0]["remaining_quantity"] == 1.0


def test_crossing_engine_fok_all_or_none():
    """Verify Fill-or-Kill orders abort when available liquidity is insufficient."""
    engine = InternalCrossingEngine()

    # Post 1.0 BTC sell
    engine.cross_order(
        InternalCrossingOrder(
            symbol="BTC-USD",
            side=CrossingSide.SELL,
            quantity=1.0,
            remaining_quantity=1.0,
            time_in_force=CrossingTimeInForce.RESTING,
        ),
        current_bid=60000.0,
        current_ask=60010.0,
    )

    # FOK order requiring 2.0 BTC -> must be rejected completely
    fok_order = InternalCrossingOrder(
        symbol="BTC-USD",
        side=CrossingSide.BUY,
        quantity=2.0,
        remaining_quantity=2.0,
        time_in_force=CrossingTimeInForce.FOK,
    )
    res_rejected = engine.cross_order(fok_order, current_bid=60000.0, current_ask=60010.0)
    assert res_rejected.filled_quantity == 0.0
    assert res_rejected.remaining_unfilled_quantity == 2.0

    # Resting sell order must remain untouched
    book = engine.get_resting_book("BTC-USD")
    assert len(book["resting_sells"]) == 1
    assert book["resting_sells"][0]["remaining_quantity"] == 1.0

    # FOK order requiring 1.0 BTC -> fills completely
    fok_success = InternalCrossingOrder(
        symbol="BTC-USD",
        side=CrossingSide.BUY,
        quantity=1.0,
        remaining_quantity=1.0,
        time_in_force=CrossingTimeInForce.FOK,
    )
    res_success = engine.cross_order(fok_success, current_bid=60000.0, current_ask=60010.0)
    assert res_success.filled_quantity == 1.0


def test_crossing_engine_limit_protection():
    """Verify buyer and seller limit prices are strictly respected against midpoint."""
    engine = InternalCrossingEngine()

    # Midpoint is (65000 + 65010) / 2 = 65005
    # Buyer limits at 65002 (unwilling to cross above 65002)
    buyer = InternalCrossingOrder(
        symbol="BTC-USD",
        side=CrossingSide.BUY,
        quantity=1.0,
        remaining_quantity=1.0,
        limit_price=65002.0,
        time_in_force=CrossingTimeInForce.IOC,
    )
    res = engine.cross_order(buyer, current_bid=65000.0, current_ask=65010.0)
    assert res.filled_quantity == 0.0


def test_crossing_engine_cancellation_and_metrics():
    """Verify order cancellation and operational metrics accumulation."""
    engine = InternalCrossingEngine()

    order = InternalCrossingOrder(
        symbol="SOL-USD",
        side=CrossingSide.BUY,
        quantity=10.0,
        remaining_quantity=10.0,
        time_in_force=CrossingTimeInForce.RESTING,
    )
    engine.cross_order(order, current_bid=150.0, current_ask=150.2)

    metrics = engine.get_metrics("SOL-USD")
    assert metrics.resting_buy_volume == 10.0
    assert metrics.resting_buy_count == 1

    # Cancel order
    cancelled = engine.cancel_order("SOL-USD", order.order_id)
    assert cancelled is True

    metrics_after = engine.get_metrics("SOL-USD")
    assert metrics_after.resting_buy_volume == 0.0
    assert metrics_after.resting_buy_count == 0
