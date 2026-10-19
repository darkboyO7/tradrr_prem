"""Tests for Institutional High-Fidelity Limit Order Book Replay & Microstructure Backtester."""

import pytest

from backtesting.orderbook_simulator import (
    MicrostructureFill,
    MicrostructureSimulationReport,
    OrderBookMicrostructureSimulator,
    QueuePriorityModel,
    SimOrderSide,
    SimOrderStatus,
    SimOrderType,
    SimulatedOrder,
)


@pytest.fixture
def btc_simulator():
    """Create a simulator instance with 5ms round-trip latency."""
    return OrderBookMicrostructureSimulator(
        symbol="BTC-USD",
        initial_cash=100_000.0,
        maker_fee_bps=-1.0,  # 1 bps rebate
        taker_fee_bps=5.0,   # 5 bps fee
        latency_ns=5_000_000, # 5 ms
        queue_priority=QueuePriorityModel.FIFO,
        market_impact_factor=0.1,
    )


def test_snapshot_and_market_order_sweep(btc_simulator):
    """Test L2 snapshot application and aggressive market order execution sweeping depth."""
    sim = btc_simulator
    t0 = 1_000_000_000

    # 1. Provide book snapshot
    # Bids: 65,000 (2.0), 64,990 (3.0)
    # Asks: 65,010 (1.5), 65,020 (4.0)
    sim.apply_snapshot(
        timestamp_ns=t0,
        bids=[(65000.0, 2.0), (64990.0, 3.0)],
        asks=[(65010.0, 1.5), (65020.0, 4.0)],
    )

    assert sim.best_bid == 65000.0
    assert sim.best_ask == 65010.0
    assert sim.mid_price == 65005.0

    # 2. Submit market BUY of 2.0 BTC (sweeps 1.5 at 65010 and 0.5 at 65020)
    order = sim.submit_order(
        order_id="ORD_001",
        side=SimOrderSide.BUY,
        order_type=SimOrderType.MARKET,
        quantity=2.0,
    )

    # Order is pending due to 5ms latency
    assert order.status == SimOrderStatus.PENDING

    # Advance time by 5ms (arrival at exchange)
    t1 = t0 + 5_000_000
    sim.apply_snapshot(
        timestamp_ns=t1,
        bids=[(65000.0, 2.0), (64990.0, 3.0)],
        asks=[(65010.0, 1.5), (65020.0, 4.0)],
    )

    assert order.status == SimOrderStatus.FILLED
    assert order.filled_quantity == 2.0
    assert len(order.fill_events) == 2
    assert sim.position == 2.0
    assert sim.cash < 100_000.0 - (2.0 * 65000.0)

    # Fills are taker fills with positive fee
    report = sim.generate_report()
    assert report.total_orders_filled == 1
    assert report.total_volume_traded == 2.0
    assert report.total_fees_paid > 0.0
    assert report.maker_fill_ratio == 0.0


def test_fifo_queue_priority_and_passive_limit_order(btc_simulator):
    """Verify FIFO queue position tracking and passive limit fill upon trade matching."""
    sim = btc_simulator
    t0 = 1_000_000_000

    # Book has 2.0 BTC at 65,000 bid
    sim.apply_snapshot(
        timestamp_ns=t0,
        bids=[(65000.0, 2.0)],
        asks=[(65010.0, 2.0)],
    )

    # Submit resting limit BUY at 65,000 for 1.0 BTC
    order = sim.submit_order(
        order_id="ORD_LIMIT_01",
        side=SimOrderSide.BUY,
        order_type=SimOrderType.LIMIT,
        quantity=1.0,
        limit_price=65000.0,
    )

    # Advance time 5ms to activate order
    sim.apply_snapshot(
        timestamp_ns=t0 + 5_000_000,
        bids=[(65000.0, 2.0)],
        asks=[(65010.0, 2.0)],
    )

    assert order.status == SimOrderStatus.ACTIVE
    # 2.0 BTC was ahead of us in FIFO queue
    assert order.queue_ahead_quantity == 2.0

    # 1. External SELL market trade of 1.5 BTC occurs at 65,000 (hits queue ahead)
    sim.apply_market_trade(
        timestamp_ns=t0 + 6_000_000,
        side=SimOrderSide.SELL,
        price=65000.0,
        quantity=1.5,
    )

    # Order should NOT be filled yet; queue ahead reduced to 0.5
    assert order.status == SimOrderStatus.ACTIVE
    assert order.filled_quantity == 0.0
    assert order.queue_ahead_quantity == pytest.approx(0.5)

    # 2. External SELL trade of 1.0 BTC occurs at 65,000 (depletes remaining 0.5 ahead + fills 0.5 of ours)
    sim.apply_market_trade(
        timestamp_ns=t0 + 7_000_000,
        side=SimOrderSide.SELL,
        price=65000.0,
        quantity=1.0,
    )

    assert order.status == SimOrderStatus.PARTIALLY_FILLED
    assert order.filled_quantity == pytest.approx(0.5)
    assert order.queue_ahead_quantity == 0.0

    # 3. Another trade fills remainder
    sim.apply_market_trade(
        timestamp_ns=t0 + 8_000_000,
        side=SimOrderSide.SELL,
        price=65000.0,
        quantity=0.5,
    )

    assert order.status == SimOrderStatus.FILLED
    assert order.filled_quantity == 1.0
    # Maker fills receive rebate (maker_fee_bps is negative, fee < 0)
    fill = sim._fills[-1]
    assert fill.is_taker is False
    assert fill.venue_fee < 0.0  # Maker rebate received


def test_queue_depletion_on_cancellations(btc_simulator):
    """Test that size cancellations at the price level deplete queue ahead."""
    sim = btc_simulator
    t0 = 1_000_000_000

    sim.apply_snapshot(
        timestamp_ns=t0,
        bids=[(65000.0, 5.0)],
        asks=[(65010.0, 2.0)],
    )

    order = sim.submit_order(
        order_id="ORD_REST",
        side=SimOrderSide.BUY,
        order_type=SimOrderType.LIMIT,
        quantity=1.0,
        limit_price=65000.0,
        custom_latency_ns=0,
    )
    assert order.queue_ahead_quantity == 5.0

    # Delta update reduces depth at 65,000 from 5.0 to 1.0 (4.0 cancelled)
    sim.apply_delta(
        timestamp_ns=t0 + 1_000_000,
        side=SimOrderSide.BUY,
        price=65000.0,
        quantity=1.0,
    )

    # With 0.5 cancellation depletion rate, 4.0 * 0.5 = 2.0 cancelled ahead
    assert order.queue_ahead_quantity == pytest.approx(3.0)


def test_pro_rata_queue_allocation():
    """Test pro-rata queue fill allocation."""
    sim = OrderBookMicrostructureSimulator(
        symbol="ETH-USD",
        queue_priority=QueuePriorityModel.PRO_RATA,
        latency_ns=0,
    )
    t0 = 100_000_000

    sim.apply_snapshot(
        timestamp_ns=t0,
        bids=[(3000.0, 10.0)],
        asks=[(3005.0, 5.0)],
    )

    # Submit 5.0 ETH bid (total level becomes 15.0; our share is 5 / 15 = 33.33%)
    order = sim.submit_order(
        order_id="ORD_PR",
        side=SimOrderSide.BUY,
        order_type=SimOrderType.LIMIT,
        quantity=5.0,
        limit_price=3000.0,
    )

    # Inbound sell trade of 6.0 ETH
    sim.apply_market_trade(
        timestamp_ns=t0 + 10_000,
        side=SimOrderSide.SELL,
        price=3000.0,
        quantity=6.0,
    )

    # We receive 6.0 * (5.0 / 15.0) = 2.0 ETH
    assert order.filled_quantity == pytest.approx(2.0)
    assert order.status == SimOrderStatus.PARTIALLY_FILLED


def test_order_cancellation_and_report_serialization(btc_simulator):
    """Test active order cancellation and comprehensive simulation reporting."""
    sim = btc_simulator
    t0 = 1_000_000_000

    sim.apply_snapshot(
        timestamp_ns=t0,
        bids=[(65000.0, 1.0)],
        asks=[(65010.0, 1.0)],
    )

    order = sim.submit_order(
        order_id="ORD_CANCEL",
        side=SimOrderSide.SELL,
        order_type=SimOrderType.LIMIT,
        quantity=0.5,
        limit_price=65050.0,
        custom_latency_ns=0,
    )
    assert order.status == SimOrderStatus.ACTIVE

    res = sim.cancel_order("ORD_CANCEL")
    assert res is True
    assert order.status == SimOrderStatus.CANCELLED

    # Re-cancelling returns False
    assert sim.cancel_order("ORD_CANCEL") is False

    report = sim.generate_report()
    assert isinstance(report, MicrostructureSimulationReport)
    rep_dict = report.to_dict()
    assert rep_dict["total_orders_submitted"] == 1
    assert rep_dict["total_orders_filled"] == 0
    assert rep_dict["symbol"] == "BTC-USD"
