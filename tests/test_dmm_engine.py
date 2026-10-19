"""Tests for Dynamic Market Maker (DMM) & Automated Quoting Surface Engine."""

import math
import pytest

from execution.market_maker import (
    DMMConfig,
    DMMQuoteLevel,
    DMMQuoteSurface,
    DynamicMarketMakerEngine,
    InventoryState,
)


def test_dmm_neutral_inventory_symmetric_quotes():
    """Verify that zero inventory produces symmetric quotes around mid-price."""
    engine = DynamicMarketMakerEngine(config=DMMConfig(gamma=0.01, quote_levels=3))
    engine.set_inventory("BTC-USD", 0.0)

    surface = engine.calculate_quotes(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        bid_volume=1.0,
        ask_volume=1.0,
        volatility_annualized=0.50,
    )

    # With q = 0, reservation price equals mid-price
    assert surface.mid_price == 65005.0
    assert surface.reservation_price == 65005.0
    assert surface.inventory_state == InventoryState.BALANCED
    assert surface.bids_active is True
    assert surface.asks_active is True
    assert len(surface.quotes) == 3

    # Multi-level quote ladder monotonicity
    for i in range(len(surface.quotes) - 1):
        curr = surface.quotes[i]
        nxt = surface.quotes[i + 1]
        assert curr.bid_price > nxt.bid_price  # Bids decrease further out
        assert curr.ask_price < nxt.ask_price  # Asks increase further out
        assert nxt.bid_size > curr.bid_size    # Outer tiers have larger size


def test_dmm_long_inventory_skewing():
    """Verify long inventory lowers reservation price to offload asset."""
    engine = DynamicMarketMakerEngine(config=DMMConfig(gamma=0.01))
    engine.set_inventory("BTC-USD", 3.0)  # Long 3 BTC

    surface = engine.calculate_quotes(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        volatility_annualized=0.50,
    )

    # Reservation price should be strictly below mid-price
    assert surface.reservation_price < surface.mid_price
    assert surface.inventory_skew_bps > 0.0
    assert surface.inventory_state == InventoryState.LONG_SKEWED

    # Ask quotes are more aggressive (tighter to mid) than bid quotes
    level_1 = surface.quotes[0]
    assert level_1.ask_spread_bps < level_1.bid_spread_bps


def test_dmm_short_inventory_skewing():
    """Verify short inventory raises reservation price to incentivize buying."""
    engine = DynamicMarketMakerEngine(config=DMMConfig(gamma=0.01))
    engine.set_inventory("BTC-USD", -3.0)  # Short 3 BTC

    surface = engine.calculate_quotes(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        volatility_annualized=0.50,
    )

    # Reservation price should be strictly above mid-price
    assert surface.reservation_price > surface.mid_price
    assert surface.inventory_state == InventoryState.SHORT_SKEWED

    # Bid quotes are more aggressive (tighter to mid) than ask quotes
    level_1 = surface.quotes[0]
    assert level_1.bid_spread_bps < level_1.ask_spread_bps


def test_dmm_inventory_ceiling_circuit_breaker():
    """Verify exceeding max inventory halts one-sided quoting to protect capital."""
    engine = DynamicMarketMakerEngine(config=DMMConfig(max_inventory_units=5.0))
    engine.set_inventory("ETH-USD", 5.0)  # Max ceiling hit

    surface = engine.calculate_quotes(
        symbol="ETH-USD",
        best_bid=3500.0,
        best_ask=3502.0,
    )

    assert surface.inventory_state == InventoryState.LIMIT_BREACHED
    assert surface.bids_active is False
    assert surface.asks_active is True

    # Bids must have size 0.0 to prevent accumulating more long exposure
    for q in surface.quotes:
        assert q.bid_size == 0.0
        assert q.ask_size > 0.0


def test_dmm_ofi_adverse_selection_defense():
    """Verify toxic order flow (OFI spike) widens spreads against informed flow."""
    engine = DynamicMarketMakerEngine(config=DMMConfig(ofi_sensitivity=0.5))
    engine.set_inventory("BTC-USD", 0.0)

    # Baseline calm quotes
    calm_surface = engine.calculate_quotes(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        order_flow_imbalance=0.0,
    )

    # Surge of aggressive buyer flow (OFI = +5.0)
    toxic_surface = engine.calculate_quotes(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        order_flow_imbalance=5.0,
    )

    assert toxic_surface.ofi_adverse_selection_bps > 0.0
    # Ask quotes are pushed further out (widened) to prevent adverse selection
    assert toxic_surface.quotes[0].ask_price > calm_surface.quotes[0].ask_price


def test_dmm_fill_recording_and_tracking():
    """Verify execution fills accurately update inventory position."""
    engine = DynamicMarketMakerEngine()
    engine.set_inventory("SOL-USD", 0.0)

    # Buy fill
    inv1 = engine.record_fill(symbol="SOL-USD", side="BUY", quantity=2.0, price=150.0)
    assert inv1 == 2.0
    assert engine.get_inventory("SOL-USD") == 2.0

    # Sell fill
    inv2 = engine.record_fill(symbol="SOL-USD", side="SELL", quantity=0.8, price=151.0)
    assert math.isclose(inv2, 1.2, abs_tol=1e-5)
    assert math.isclose(engine.get_inventory("SOL-USD"), 1.2, abs_tol=1e-5)
