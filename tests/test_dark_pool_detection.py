"""Tests for Real-Time Dark Pool & Hidden Liquidity Detection Engine."""

from datetime import datetime, timedelta, timezone
import pytest
from analysis.dark_pool_detection import (
    DarkPoolDetectionEngine,
    DarkPoolDetectionReport,
    DarkPrintRecord,
    HiddenLiquidityType,
    IcebergSignature,
)


def test_dark_cross_midpoint_trade_print_detection():
    """Verify trades executed at midpoint inside NBBO spread are flagged as DARK_CROSS_MIDPOINT."""
    engine = DarkPoolDetectionEngine()
    now = datetime.now(timezone.utc)

    # NBBO: 65,000.00 / 65,020.00 -> midpoint = 65,010.00
    record = engine.process_trade_print(
        symbol="BTC-USD",
        venue="coinbase",
        price=65010.0,
        quantity=2.0,
        best_bid=65000.0,
        best_ask=65020.0,
        timestamp=now,
    )

    assert record is not None
    assert record.symbol == "BTC-USD"
    assert record.venue == "coinbase"
    assert record.liquidity_type == HiddenLiquidityType.DARK_CROSS_MIDPOINT
    assert record.notional_value_usd == 130020.0
    assert record.spread_bps_at_time > 0.0
    assert record.price_improvement_bps > 0.0


def test_off_book_internalization_detection():
    """Verify trades executed inside spread away from midpoint are flagged as OFF_BOOK_INTERNALIZATION."""
    engine = DarkPoolDetectionEngine()
    now = datetime.now(timezone.utc)

    # Trade printed at 65,004.0 (spread is 65,000.0 to 65,020.0, midpoint is 65,010.0)
    record = engine.process_trade_print(
        symbol="BTC-USD",
        venue="kraken",
        price=65004.0,
        quantity=1.5,
        best_bid=65000.0,
        best_ask=65020.0,
        timestamp=now,
    )

    assert record is not None
    assert record.liquidity_type == HiddenLiquidityType.OFF_BOOK_INTERNALIZATION
    assert record.price_improvement_bps > 0.0


def test_iceberg_order_replenishment_detection():
    """Verify repeated depth replenishments at a price level identify an active iceberg order."""
    engine = DarkPoolDetectionEngine(replenishment_window_seconds=2.0, min_replenishments=2)
    t0 = datetime.now(timezone.utc)

    # 1. Initial visible bid depth: 5.0 BTC @ $65,000
    engine.process_book_depth_update("BTC-USD", "binance", "BUY", 65000.0, 5.0, timestamp=t0)

    # 2. Trade exhausts tranche: depth drops to 1.0 BTC
    t1 = t0 + timedelta(milliseconds=200)
    engine.process_book_depth_update("BTC-USD", "binance", "BUY", 65000.0, 1.0, timestamp=t1)

    # 3. Replenishment 1: depth pops back up to 5.0 BTC
    t2 = t1 + timedelta(milliseconds=300)
    sig1 = engine.process_book_depth_update("BTC-USD", "binance", "BUY", 65000.0, 5.0, timestamp=t2)
    assert sig1 is None  # Needs 2 replenishments

    # 4. Another trade drops depth to 1.5 BTC
    t3 = t2 + timedelta(milliseconds=200)
    engine.process_book_depth_update("BTC-USD", "binance", "BUY", 65000.0, 1.5, timestamp=t3)

    # 5. Replenishment 2: depth pops back up to 5.0 BTC
    t4 = t3 + timedelta(milliseconds=250)
    sig2 = engine.process_book_depth_update("BTC-USD", "binance", "BUY", 65000.0, 5.0, timestamp=t4)

    assert sig2 is not None
    assert sig2.symbol == "BTC-USD"
    assert sig2.venue == "binance"
    assert sig2.side == "BUY"
    assert sig2.price_level == 65000.0
    assert sig2.replenishment_count == 2
    assert sig2.confidence_score >= 0.70
    assert sig2.estimated_hidden_qty > 0.0
    assert sig2.is_active is True

    # Active icebergs query
    active = engine.detect_icebergs("BTC-USD")
    assert len(active) == 1
    assert active[0].signature_id == sig2.signature_id


def test_dark_pool_detection_report_aggregation():
    """Verify aggregated metrics for lit vs. dark volume share and market impact implications."""
    engine = DarkPoolDetectionEngine()
    now = datetime.now(timezone.utc)

    # Lit trade at the ask
    engine.process_trade_print("ETH-USD", "coinbase", price=3500.0, quantity=10.0, best_bid=3499.0, best_ask=3500.0, timestamp=now)
    # Dark midpoint trade
    engine.process_trade_print("ETH-USD", "coinbase", price=3499.5, quantity=5.0, best_bid=3499.0, best_ask=3500.0, timestamp=now)

    report = engine.generate_report("ETH-USD")
    assert report.symbol == "ETH-USD"
    assert report.total_lit_volume_usd == 35000.0
    assert report.total_dark_volume_usd == 3499.5 * 5.0
    assert 0.0 < report.dark_volume_share_pct < 100.0
    assert len(report.recent_dark_prints) == 1
    assert report.market_impact_implication in ["BALANCED", "HIGH_DARK_CROSS_ACTIVITY", "HEAVY_BUYING_ABSORPTION", "HEAVY_SELLING_ABSORPTION"]


def test_clear_stale_signatures():
    """Verify inactive/expired iceberg signatures are deactivated."""
    engine = DarkPoolDetectionEngine()
    old_time = datetime.now(timezone.utc) - timedelta(seconds=120)

    # Seed an iceberg with old timestamp
    engine.process_book_depth_update("SOL-USD", "binance", "SELL", 150.0, 50.0, timestamp=old_time)
    engine.process_book_depth_update("SOL-USD", "binance", "SELL", 150.0, 10.0, timestamp=old_time + timedelta(milliseconds=100))
    engine.process_book_depth_update("SOL-USD", "binance", "SELL", 150.0, 50.0, timestamp=old_time + timedelta(milliseconds=200))
    engine.process_book_depth_update("SOL-USD", "binance", "SELL", 150.0, 10.0, timestamp=old_time + timedelta(milliseconds=300))
    sig = engine.process_book_depth_update("SOL-USD", "binance", "SELL", 150.0, 50.0, timestamp=old_time + timedelta(milliseconds=400))
    assert sig is not None
    assert sig.is_active is True

    # Clear stale
    deactivated = engine.clear_stale_signatures(max_age_seconds=60.0)
    assert deactivated == 1
    assert len(engine.detect_icebergs("SOL-USD")) == 0
