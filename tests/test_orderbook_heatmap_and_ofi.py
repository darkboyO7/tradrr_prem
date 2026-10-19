"""Tests for Order Flow Imbalance (OFI), Liquidity Heatmap, and Queue Position Estimator."""

from datetime import datetime, timezone
import pytest

from data.orderbook.heatmap import (
    BBOState,
    DepthHeatmapGrid,
    OFIMetric,
    OrderBookHeatmapEngine,
    OrderFlowImbalanceCalculator,
    QueuePositionEstimator,
    QueuePositionReport,
)


def test_ofi_calculation_cases():
    calc = OrderFlowImbalanceCalculator(rolling_window=10, z_threshold=1.5)

    # Initial state
    t0 = datetime.now(timezone.utc)
    bbo_0 = BBOState(bid_price=65000.0, bid_size=2.0, ask_price=65010.0, ask_size=3.0, timestamp=t0)
    m0 = calc.compute_step(bbo_0)
    assert m0.instantaneous_ofi == 0.0

    # Case 1: Bid price rises (aggressive buying interest)
    # curr.bid_price > prev.bid_price => I_b = curr.bid_size (4.0)
    # ask unchanged => I_a = 0
    bbo_1 = BBOState(bid_price=65002.0, bid_size=4.0, ask_price=65010.0, ask_size=3.0, timestamp=t0)
    m1 = calc.compute_step(bbo_1)
    assert m1.bid_flow == 4.0
    assert m1.ask_flow == 0.0
    assert m1.instantaneous_ofi == 4.0

    # Case 2: Bid price equal, size increases (+2.5)
    bbo_2 = BBOState(bid_price=65002.0, bid_size=6.5, ask_price=65010.0, ask_size=3.0, timestamp=t0)
    m2 = calc.compute_step(bbo_2)
    assert m2.bid_flow == 2.5
    assert m2.instantaneous_ofi == 2.5

    # Case 3: Bid price drops (liquidity withdrawal / fill)
    # curr.bid_price < prev.bid_price => I_b = -prev.bid_size (-6.5)
    bbo_3 = BBOState(bid_price=65000.0, bid_size=1.0, ask_price=65010.0, ask_size=3.0, timestamp=t0)
    m3 = calc.compute_step(bbo_3)
    assert m3.bid_flow == -6.5
    assert m3.instantaneous_ofi == -6.5

    # Case 4: Ask price drops (aggressive selling pressure)
    # curr.ask_price < prev.ask_price => I_a = -curr.ask_size (-5.0) => OFI = -(-5.0) = +5.0 ? No:
    # Cont-Kukanov-Stoikov: I_a = -curr.ask_size (-5.0) => OFI = I_b - I_a = 0 - (-5.0) = +5.0 or downward?
    # Note: when ask drops, sellers push price down: OFI = 0 - (-5.0) = -(-5.0)
    bbo_4 = BBOState(bid_price=65000.0, bid_size=1.0, ask_price=65008.0, ask_size=5.0, timestamp=t0)
    m4 = calc.compute_step(bbo_4)
    assert m4.ask_flow == -5.0


def test_multi_level_ofi_weighting():
    calc = OrderFlowImbalanceCalculator()

    prev_bids = [(65000.0, 2.0), (64995.0, 3.0), (64990.0, 5.0)]
    curr_bids = [(65002.0, 3.0), (64995.0, 4.0), (64990.0, 2.0)]

    # Level 1: price up -> flow = 3.0 * (1/1) = 3.0
    # Level 2: price equal -> flow = (4.0 - 3.0) * (1/2) = 0.5
    # Level 3: price equal -> flow = (2.0 - 5.0) * (1/3) = -1.0
    # Expected total = 3.0 + 0.5 - 1.0 = 2.5
    multi_ofi = calc.compute_multi_level_ofi(prev_bids, curr_bids, is_bid=True, depth=3)
    assert multi_ofi == pytest.approx(2.5, abs=1e-3)


def test_heatmap_engine_grid_and_gini():
    engine = OrderBookHeatmapEngine(symbol="BTC-USD", max_snapshots=20)

    # Record 5 depth snapshots
    for i in range(5):
        bids = [(65000.0 - (j * 10.0), 1.0 + j * 0.5) for j in range(5)]
        asks = [(65010.0 + (j * 10.0), 1.0 + j * 0.5) for j in range(5)]
        engine.record_snapshot(bids=bids, asks=asks)

    grid = engine.generate_heatmap(num_price_levels=15)

    assert isinstance(grid, DepthHeatmapGrid)
    assert grid.symbol == "BTC-USD"
    assert len(grid.price_levels) == 15
    assert len(grid.time_labels) == 5
    assert len(grid.bid_intensity_grid) == 5
    assert len(grid.ask_intensity_grid) == 5
    assert 0.0 <= grid.liquidity_concentration_index <= 1.0


def test_queue_position_estimator_drain_and_fill():
    estimator = QueuePositionEstimator(
        order_id="ord_queue_001",
        symbol="BTC-USD",
        price=65000.0,
        order_size=2.0,
        initial_depth_ahead=5.0,
    )

    # 1. Trade at same price drains 3.0 of queue ahead (2.0 left)
    rep1 = estimator.process_trade(trade_price=65000.0, trade_size=3.0)
    assert rep1.current_queue_ahead == 2.0
    assert rep1.filled_quantity == 0.0
    assert rep1.is_filled is False
    assert rep1.fill_percentage == 0.0

    # 2. Trade of 3.0 drains remaining 2.0 queue and fills 1.0 of our order
    rep2 = estimator.process_trade(trade_price=65000.0, trade_size=3.0)
    assert rep2.current_queue_ahead == 0.0
    assert rep2.filled_quantity == 1.0
    assert rep2.fill_percentage == 50.0
    assert rep2.is_filled is False

    # 3. Trade of 2.0 fills remaining 1.0 of our order
    rep3 = estimator.process_trade(trade_price=65000.0, trade_size=2.0)
    assert rep3.filled_quantity == 2.0
    assert rep3.fill_percentage == 100.0
    assert rep3.is_filled is True
    assert rep3.estimated_fill_probability == 1.0

    # 4. Trade at a different price leaves order unchanged
    rep4 = estimator.process_trade(trade_price=65010.0, trade_size=10.0)
    assert rep4.is_filled is True
