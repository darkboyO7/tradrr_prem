"""Tests for Level 2 Order Book & Microstructure Analytics."""

import pytest
from data.orderbook.l2_book import OrderBookL2, OrderBookMetrics


def test_orderbook_snapshot_and_top_levels():
    """Verify L2 order book applies snapshot and maintains sorted levels."""
    book = OrderBookL2(symbol="BTC-USD", max_depth_levels=10)

    bids = [(65000.0, 1.5), (64990.0, 3.0), (65005.0, 0.8)]
    asks = [(65010.0, 1.2), (65020.0, 2.5), (65015.0, 0.5)]

    book.apply_snapshot(bids=bids, asks=asks)

    assert book.best_bid == 65005.0
    assert book.best_ask == 65010.0
    assert book.spread == 5.0
    assert book.mid_price == 65007.5


def test_orderbook_delta_updates():
    """Verify incremental delta additions, updates, and level removals."""
    book = OrderBookL2(symbol="BTC-USD")
    book.apply_snapshot(
        bids=[(65000.0, 1.0), (64990.0, 2.0)],
        asks=[(65010.0, 1.0), (65020.0, 2.0)],
    )

    # Apply deltas: insert new best bid, update ask, remove level (qty=0)
    delta_bids = [(65005.0, 2.5), (64990.0, 0.0)]  # remove 64990
    delta_asks = [(65010.0, 3.5)]  # update qty

    book.apply_delta(bids=delta_bids, asks=delta_asks)

    assert book.best_bid == 65005.0
    assert 64990.0 not in book._bids
    assert book._asks[65010.0] == 3.5


def test_orderbook_microstructure_metrics():
    """Verify quantitative micro-price and Order Book Imbalance (OBI) calculations."""
    book = OrderBookL2(symbol="ETH-USD")
    # Heavy bid volume should pull micro-price closer to ask and produce positive OBI
    bids = [(3500.0, 10.0), (3499.0, 5.0)]
    asks = [(3502.0, 2.0), (3503.0, 3.0)]

    book.apply_snapshot(bids=bids, asks=asks)
    metrics = book.compute_metrics(depth_levels=2)

    assert isinstance(metrics, OrderBookMetrics)
    assert metrics.best_bid == 3500.0
    assert metrics.best_ask == 3502.0
    assert metrics.spread == 2.0
    assert metrics.spread_bps > 0
    # Volume-weighted micro-price should exceed mid-price due to bid dominance
    assert metrics.micro_price > metrics.mid_price
    # OBI: (15 - 5) / (15 + 5) = 10 / 20 = 0.5
    assert metrics.order_book_imbalance == 0.5
