"""Tests for Smart Order Router (SOR) and Multi-Venue Arbitrage Scanner."""

from unittest.mock import AsyncMock
import pytest

from analysis.arbitrage import CrossExchangeArbitrageScanner
from data.orderbook.consolidated import ArbitrageOpportunity, ConsolidatedOrderBook
from execution.router import ChildOrderSlice, RoutingPlan, RoutingPolicy, SmartOrderRouter


def test_sor_best_price_routing():
    """Verify BEST_PRICE policy routes BUY to lowest ask and SELL to highest bid."""
    cob = ConsolidatedOrderBook("BTC-USD")
    cob.update_venue_snapshot("binance", bids=[(64000.0, 5.0)], asks=[(64050.0, 5.0)])
    cob.update_venue_snapshot("kraken", bids=[(64010.0, 5.0)], asks=[(64040.0, 5.0)])  # Lowest ask 64040, highest bid 64010
    cob.update_venue_snapshot("coinbase", bids=[(63990.0, 5.0)], asks=[(64060.0, 5.0)])

    router = SmartOrderRouter(consolidated_book=cob)

    # 1. Buy 2.0 BTC
    plan_buy = router.plan_order("BTC-USD", side="BUY", quantity=2.0, policy=RoutingPolicy.BEST_PRICE)
    assert plan_buy.policy == RoutingPolicy.BEST_PRICE
    assert len(plan_buy.slices) == 1
    assert plan_buy.slices[0].venue == "kraken"
    assert plan_buy.slices[0].expected_price == 64040.0
    assert plan_buy.slices[0].quantity == 2.0

    # 2. Sell 1.5 BTC
    plan_sell = router.plan_order("BTC-USD", side="SELL", quantity=1.5, policy=RoutingPolicy.BEST_PRICE)
    assert len(plan_sell.slices) == 1
    assert plan_sell.slices[0].venue == "kraken"
    assert plan_sell.slices[0].expected_price == 64010.0
    assert plan_sell.slices[0].quantity == 1.5


def test_sor_fee_aware_routing():
    """Verify FEE_AWARE policy chooses venue with lowest net effective cost after venue fees."""
    cob = ConsolidatedOrderBook("BTC-USD")
    # Venue A (Coinbase): Ask = 65000.0, but fee = 20 bps (0.0020) -> net = 65000 * 1.002 = 65130.0
    cob.update_venue_snapshot("coinbase", bids=[(64900.0, 5.0)], asks=[(65000.0, 5.0)])

    # Venue B (Binance): Ask = 65020.0, but fee = 5 bps (0.0005) -> net = 65020 * 1.0005 = 65052.51
    cob.update_venue_snapshot("binance", bids=[(64910.0, 5.0)], asks=[(65020.0, 5.0)])

    router = SmartOrderRouter(
        consolidated_book=cob,
        venue_fee_rates={"coinbase": 0.0020, "binance": 0.0005},
    )

    # Raw best price is Coinbase (65000.0 < 65020.0)
    best_raw = router.plan_order("BTC-USD", side="BUY", quantity=1.0, policy=RoutingPolicy.BEST_PRICE)
    assert best_raw.slices[0].venue == "coinbase"

    # Fee aware router should select Binance due to lower all-in cost!
    fee_aware = router.plan_order("BTC-USD", side="BUY", quantity=1.0, policy=RoutingPolicy.FEE_AWARE)
    assert fee_aware.slices[0].venue == "binance"
    assert fee_aware.slices[0].expected_price == 65020.0


def test_sor_split_depth_liquidity_slicing():
    """Verify SPLIT_DEPTH distributes a large parent order proportional to available depth."""
    cob = ConsolidatedOrderBook("ETH-USD")
    # Binance top depth: 30.0 ETH (75% of total depth)
    cob.update_venue_snapshot("binance", bids=[(3500.0, 20.0)], asks=[(3510.0, 30.0)])
    # Kraken top depth: 10.0 ETH (25% of total depth)
    cob.update_venue_snapshot("kraken", bids=[(3498.0, 20.0)], asks=[(3512.0, 10.0)])

    router = SmartOrderRouter(consolidated_book=cob)

    # Route 20.0 ETH parent order across venues
    plan = router.plan_order("ETH-USD", side="BUY", quantity=20.0, policy=RoutingPolicy.SPLIT_DEPTH)

    assert len(plan.slices) == 2
    venue_slices = {s.venue: s.quantity for s in plan.slices}
    # 75% of 20 = 15.0 to Binance, 25% of 20 = 5.0 to Kraken
    assert venue_slices["binance"] == pytest.approx(15.0, 0.01)
    assert venue_slices["kraken"] == pytest.approx(5.0, 0.01)
    # Sum of slices must equal parent quantity exactly
    assert sum(s.quantity for s in plan.slices) == pytest.approx(20.0, 1e-6)


@pytest.mark.asyncio
async def test_sor_execute_plan_mock_clients():
    """Verify asynchronous execution of child order slices via registered clients."""
    cob = ConsolidatedOrderBook("BTC-USD")
    cob.update_venue_snapshot("binance", bids=[(65000.0, 5.0)], asks=[(65010.0, 5.0)])
    cob.update_venue_snapshot("kraken", bids=[(64995.0, 5.0)], asks=[(65020.0, 5.0)])

    mock_binance = AsyncMock()
    mock_binance.place_order = AsyncMock(
        return_value={"executed_quantity": 1.0, "price": 65010.0, "status": "FILLED"}
    )

    router = SmartOrderRouter(consolidated_book=cob, venue_clients={"binance": mock_binance})

    plan = router.plan_order("BTC-USD", side="BUY", quantity=1.0, policy=RoutingPolicy.BEST_PRICE)
    res = await router.execute_plan(plan)

    assert res["status"] == "COMPLETED"
    assert res["total_filled_quantity"] == 1.0
    assert res["realized_vwap"] == 65010.0
    assert mock_binance.place_order.called


@pytest.mark.asyncio
async def test_sor_execute_arbitrage_dual_leg():
    """Verify concurrent dual-leg sweep execution for cross-exchange arbitrage."""
    cob = ConsolidatedOrderBook("SOL-USD")
    opp = ArbitrageOpportunity(
        symbol="SOL-USD",
        buy_venue="binance",
        buy_price=140.0,
        buy_size=10.0,
        sell_venue="kraken",
        sell_price=145.0,
        sell_size=10.0,
        max_executable_volume=10.0,
        gross_spread=5.0,
        gross_spread_bps=350.0,
        net_spread=4.75,
        net_profit_bps=330.0,
        estimated_net_pnl=47.50,
    )

    router = SmartOrderRouter(consolidated_book=cob)
    res = await router.execute_arbitrage(opp, volume=5.0)

    assert res["status"] == "COMPLETED"
    assert res["total_filled_quantity"] == 10.0  # 5 buy + 5 sell
    assert res["policy"] == RoutingPolicy.ARBITRAGE_SWEEP.value
    assert len(res["slices"]) == 2
    assert res["net_profit_bps_expected"] == 330.0


def test_cross_exchange_arbitrage_scanner_telemetry():
    """Verify CrossExchangeArbitrageScanner monitors books, registers opportunities, and tracks stats."""
    cob = ConsolidatedOrderBook("BTC-USD")
    cob.update_venue_snapshot("kraken", bids=[(66000.0, 2.0)], asks=[(66100.0, 2.0)])
    cob.update_venue_snapshot("binance", bids=[(64900.0, 2.0)], asks=[(65000.0, 2.0)])

    scanner = CrossExchangeArbitrageScanner(min_net_profit_bps=5.0, min_notional_usd=500.0)
    scanner.register_book("BTC-USD", cob)

    # Scan BTC-USD
    opp = scanner.scan_symbol("BTC-USD")
    assert opp is not None
    assert opp.buy_venue == "binance"
    assert opp.sell_venue == "kraken"

    stats = scanner.get_stats()
    assert stats.total_detected == 1
    assert stats.active_opportunities_count == 1
    assert stats.total_theoretical_pnl > 0
    assert "binance->kraken" in stats.venue_pair_counts
