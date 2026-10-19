"""Tests for Real-Time Cross-Exchange Triangular & Synthetic Latency Arbitrage Engine."""

from datetime import datetime, timezone
import pytest
from analysis.triangular_arbitrage import (
    RateGraphData,
    RateGraphEdge,
    TriangularArbitrageEngine,
    TriangularArbitrageOpportunity,
    TriangularLeg,
    TriangularScanReport,
)


def test_symbol_parsing_and_quote_updates():
    """Verify symbol parsing and directed quote ingestion."""
    engine = TriangularArbitrageEngine()
    assert engine.parse_symbol("BTC-USD") == ("BTC", "USD")
    assert engine.parse_symbol("ETH/BTC") == ("ETH", "BTC")
    assert engine.parse_symbol("SOL-USDT") == ("SOL", "USDT")

    engine.update_quote("binance", "BTC-USD", bid=65000.0, ask=65005.0, bid_qty=2.0, ask_qty=2.0)
    assert "BTC" in engine._currencies
    assert "USD" in engine._currencies
    assert ("binance", "BTC-USD") in engine._quotes


def test_directed_rate_graph_construction():
    """Verify conversion of bidirectional quotes into directed rate graph edges."""
    engine = TriangularArbitrageEngine(default_slippage_bps=2.0)
    engine.update_quote("coinbase", "BTC-USD", bid=60000.0, ask=60000.0)

    graph = engine.get_rate_graph()
    assert "BTC" in graph.nodes
    assert "USD" in graph.nodes

    # There should be two directed edges: USD -> BTC (buy) and BTC -> USD (sell)
    assert len(graph.edges) == 2
    edge_buy = next(e for e in graph.edges if e.from_asset == "USD" and e.to_asset == "BTC")
    edge_sell = next(e for e in graph.edges if e.from_asset == "BTC" and e.to_asset == "USD")

    assert edge_buy.action == "BUY"
    assert abs(edge_buy.rate - (1.0 / 60000.0)) < 1e-8
    assert edge_buy.effective_rate < edge_buy.rate  # Friction deducted
    assert edge_buy.weight > 0.0

    assert edge_sell.action == "SELL"
    assert abs(edge_sell.rate - 60000.0) < 1e-4


def test_profitable_intra_venue_triangular_arbitrage():
    """Verify detection and sizing of a profitable triangular arbitrage loop on a single venue."""
    engine = TriangularArbitrageEngine(default_slippage_bps=2.0, min_net_profit_bps=5.0)

    # Synthetic loop: USD -> BTC -> ETH -> USD
    # 1. USD -> BTC @ 60,000 ask (1 USD buys 1/60000 BTC)
    engine.update_quote("binance", "BTC-USD", bid=59990.0, ask=60000.0, bid_qty=5.0, ask_qty=5.0)
    # 2. BTC -> ETH @ 0.050 ask (1 BTC buys 20 ETH)
    engine.update_quote("binance", "ETH-BTC", bid=0.049, ask=0.050, bid_qty=100.0, ask_qty=100.0)
    # 3. ETH -> USD @ 3200 bid (1 ETH sells for 3200 USD)
    # Gross product = (1/60000) * (1/0.050) * 3200 = 64,000 / 60,000 = 1.0667 (+6.67%)
    engine.update_quote("binance", "ETH-USD", bid=3200.0, ask=3205.0, bid_qty=50.0, ask_qty=50.0)

    report = engine.scan_triangular_arbitrage(anchor_asset="USD")
    assert report.opportunities_count >= 1
    assert report.best_opportunity is not None

    best = report.best_opportunity
    assert best.triangle_path == ["USD", "BTC", "ETH", "USD"]
    assert best.is_cross_venue is False
    assert best.venues_involved == ["binance"]
    assert best.gross_return_pct > 6.0
    assert best.net_return_bps > 500.0
    assert best.expected_net_profit_usd > 0.0
    assert len(best.legs) == 3
    assert best.execution_urgency == "CRITICAL"


def test_cross_venue_synthetic_arbitrage():
    """Verify multi-venue synthetic triangular arbitrage discovery."""
    engine = TriangularArbitrageEngine(default_slippage_bps=1.0, min_net_profit_bps=5.0)

    # Leg 1 on Coinbase: USD -> BTC
    engine.update_quote("coinbase", "BTC-USD", bid=60000.0, ask=60000.0, ask_qty=2.0)
    # Leg 2 on Binance: BTC -> ETH
    engine.update_quote("binance", "ETH-BTC", bid=0.048, ask=0.049, ask_qty=50.0)
    # Leg 3 on Kraken: ETH -> USD
    engine.update_quote("kraken", "ETH-USD", bid=3150.0, ask=3160.0, bid_qty=40.0)

    # Gross: (1 / 60000) * (1 / 0.049) * 3150 = 3150 / 2940 = 1.0714 (+7.14%)
    report = engine.scan_triangular_arbitrage(anchor_asset="USD")
    assert report.opportunities_count >= 1
    best = report.best_opportunity
    assert best.is_cross_venue is True
    assert set(best.venues_involved) == {"coinbase", "binance", "kraken"}


def test_non_profitable_triangle_filtered():
    """Verify non-profitable market conditions are properly filtered out."""
    engine = TriangularArbitrageEngine(min_net_profit_bps=5.0)

    # Fair non-arbitrageable prices:
    # BTC = 60,000, ETH = 3,000 -> ETH/BTC = 3000/60000 = 0.05
    engine.update_quote("binance", "BTC-USD", bid=59990.0, ask=60010.0)
    engine.update_quote("binance", "ETH-BTC", bid=0.0499, ask=0.0501)
    engine.update_quote("binance", "ETH-USD", bid=2995.0, ask=3005.0)

    report = engine.scan_triangular_arbitrage(anchor_asset="USD")
    assert report.opportunities_count == 0
    assert report.best_opportunity is None


def test_depth_and_capital_capping():
    """Verify max executable capital is constrained by shallowest leg depth."""
    engine = TriangularArbitrageEngine(max_capital_cap_usd=500000.0)

    # Shallow BTC depth: only 0.1 BTC ($6,000) available at ask
    engine.update_quote("binance", "BTC-USD", bid=60000.0, ask=60000.0, ask_qty=0.1)
    engine.update_quote("binance", "ETH-BTC", bid=0.050, ask=0.050, ask_qty=100.0)
    engine.update_quote("binance", "ETH-USD", bid=3300.0, ask=3300.0, bid_qty=100.0)

    report = engine.scan_triangular_arbitrage(anchor_asset="USD")
    assert report.best_opportunity is not None
    # Max capital should be capped near 0.1 * 60,000 = $6,000
    assert report.best_opportunity.max_executable_capital_usd <= 6005.0
