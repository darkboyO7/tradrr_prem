"""Tests for Consolidated Order Book, Synthetic BBO, and Arbitrage Detection."""

from datetime import datetime, timezone
import pytest

from data.orderbook.consolidated import ArbitrageOpportunity, ConsolidatedOrderBook, SyntheticBBO


def test_consolidated_orderbook_registration_and_bbo():
    """Verify multi-venue snapshot registration and synthetic BBO calculation."""
    cob = ConsolidatedOrderBook(symbol="BTC-USD")

    # Venue 1: Binance
    # Best Bid: 65,000, Best Ask: 65,010
    cob.update_venue_snapshot(
        venue="binance",
        bids=[(65000.0, 2.5), (64990.0, 4.0)],
        asks=[(65010.0, 1.8), (65020.0, 3.2)],
    )

    # Venue 2: Kraken
    # Best Bid: 64,995, Best Ask: 65,005 (better ask than Binance)
    cob.update_venue_snapshot(
        venue="kraken",
        bids=[(64995.0, 3.0), (64985.0, 5.0)],
        asks=[(65005.0, 2.0), (65015.0, 4.0)],
    )

    # Venue 3: Coinbase
    # Best Bid: 65,002 (better bid than Binance), Best Ask: 65,012
    cob.update_venue_snapshot(
        venue="coinbase",
        bids=[(65002.0, 1.5), (64992.0, 2.0)],
        asks=[(65012.0, 2.5), (65022.0, 3.0)],
    )

    assert set(cob.venues) == {"binance", "kraken", "coinbase"}

    bbo = cob.get_consolidated_bbo()
    assert bbo.symbol == "BTC-USD"
    # Highest bid across venues is Coinbase at 65002.0
    assert bbo.best_bid == 65002.0
    assert bbo.best_bid_venue == "coinbase"
    assert bbo.best_bid_size == 1.5

    # Lowest ask across venues is Kraken at 65005.0
    assert bbo.best_ask == 65005.0
    assert bbo.best_ask_venue == "kraken"
    assert bbo.best_ask_size == 2.0

    # Spread: 65005 - 65002 = 3.0
    assert bbo.spread == 3.0
    assert bbo.spread_bps > 0
    assert bbo.is_crossed is False


def test_consolidated_aggregated_depth_sorting():
    """Verify aggregated depth levels across venues are properly sorted."""
    cob = ConsolidatedOrderBook(symbol="ETH-USD")
    cob.update_venue_snapshot("binance", bids=[(3400.0, 10.0)], asks=[(3410.0, 5.0)])
    cob.update_venue_snapshot("kraken", bids=[(3405.0, 8.0)], asks=[(3408.0, 12.0)])

    depth = cob.get_aggregated_depth(levels=5)
    bids = depth["bids"]
    asks = depth["asks"]

    # Highest bid first: 3405.0 (Kraken), then 3400.0 (Binance)
    assert bids[0][0] == 3405.0
    assert bids[0][2] == "kraken"
    assert bids[1][0] == 3400.0
    assert bids[1][2] == "binance"

    # Lowest ask first: 3408.0 (Kraken), then 3410.0 (Binance)
    assert asks[0][0] == 3408.0
    assert asks[0][2] == "kraken"
    assert asks[1][0] == 3410.0
    assert asks[1][2] == "binance"


def test_cross_exchange_arbitrage_detection():
    """Verify crossed book detection when Venue A Bid > Venue B Ask accounting for fees."""
    cob = ConsolidatedOrderBook(symbol="SOL-USD")

    # Kraken has high bid: 155.00
    cob.update_venue_snapshot("kraken", bids=[(155.0, 50.0)], asks=[(156.0, 40.0)])

    # Binance has low ask: 150.00 (Crossed! Kraken Bid 155 > Binance Ask 150)
    cob.update_venue_snapshot("binance", bids=[(149.0, 30.0)], asks=[(150.0, 25.0)])

    bbo = cob.get_consolidated_bbo()
    assert bbo.is_crossed is True
    assert bbo.best_bid == 155.0
    assert bbo.best_ask == 150.0

    # Detect arbitrage opportunity with 0.1% taker fee
    opp = cob.detect_arbitrage(taker_fee_pct=0.001, min_net_profit_bps=5.0)
    assert opp is not None
    assert opp.symbol == "SOL-USD"
    assert opp.buy_venue == "binance"
    assert opp.buy_price == 150.0
    assert opp.sell_venue == "kraken"
    assert opp.sell_price == 155.0
    assert opp.max_executable_volume == 25.0  # min(50.0, 25.0)

    # Gross spread = 155 - 150 = 5.0
    assert opp.gross_spread == 5.0
    # Net profit must be positive and significant
    assert opp.net_spread > 4.5
    assert opp.estimated_net_pnl > 100.0


def test_cross_exchange_arbitrage_filtering_unprofitable_after_fees():
    """Verify narrow gross spreads that are eaten up by fees are filtered out."""
    cob = ConsolidatedOrderBook(symbol="BTC-USD")
    # Gross spread is only $1.00 on $65,000 (~0.15 bps)
    cob.update_venue_snapshot("kraken", bids=[(65001.0, 1.0)], asks=[(65005.0, 1.0)])
    cob.update_venue_snapshot("binance", bids=[(64998.0, 1.0)], asks=[(65000.0, 1.0)])

    # Kraken bid 65001 > Binance ask 65000, but 0.1% fee on each side is ~$130!
    opp = cob.detect_arbitrage(taker_fee_pct=0.001, min_net_profit_bps=2.0)
    assert opp is None
