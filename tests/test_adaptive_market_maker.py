"""Tests for High-Frequency Adaptive Statistical Market Making & Inventory Skew Optimizer."""

import pytest

from execution.adaptive_market_maker import (
    ASQuotingRegime,
    AdaptiveMarketMakingOptimizer,
    AdaptiveMMConfig,
    AdaptiveMMQuotingReport,
    AdaptiveQuoteTier,
)


def test_neutral_inventory_quoting():
    """Verify that with zero inventory, reservation price matches mid-price and spreads are symmetrical."""
    optimizer = AdaptiveMarketMakingOptimizer()
    optimizer.set_inventory("BTC-USD", 0.0)

    report = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        volatility_annualized=0.50,
        drift_annualized=0.0,
    )

    assert isinstance(report, AdaptiveMMQuotingReport)
    assert report.symbol == "BTC-USD"
    assert report.mid_price == 65005.0
    assert report.current_inventory == 0.0
    # Zero inventory skew
    assert report.inventory_skew_usd == 0.0
    assert report.reservation_price == 65005.0
    assert report.market_regime == ASQuotingRegime.NORMAL_TWO_SIDED
    assert report.bids_allowed is True
    assert report.asks_allowed is True
    assert len(report.quoting_tiers) == 5

    # Check tier 1 symmetry
    t1 = report.quoting_tiers[0]
    assert t1.tier_index == 1
    assert t1.bid_size == t1.ask_size
    assert t1.bid_fill_probability > 0.0
    assert t1.ask_fill_probability > 0.0


def test_inventory_skew_dynamics_long_and_short():
    """Verify that long inventory depresses reservation price and widens bids / tightens asks."""
    optimizer = AdaptiveMarketMakingOptimizer(AdaptiveMMConfig(gamma=0.01))

    # Long inventory: +3.0 BTC
    optimizer.set_inventory("BTC-USD", 3.0)
    report_long = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        volatility_annualized=0.50,
    )

    # Reservation price should be strictly below mid price to deter buying and encourage selling
    assert report_long.reservation_price < report_long.mid_price
    assert report_long.inventory_skew_usd > 0.0
    assert report_long.market_regime == ASQuotingRegime.INVENTORY_FLATTENING

    # Sizing skew: bids smaller than asks
    t1_long = report_long.quoting_tiers[0]
    assert t1_long.bid_size < t1_long.ask_size

    # Short inventory: -3.0 BTC
    optimizer.set_inventory("BTC-USD", -3.0)
    report_short = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        volatility_annualized=0.50,
    )

    assert report_short.reservation_price > report_short.mid_price
    assert report_short.inventory_skew_usd < 0.0
    t1_short = report_short.quoting_tiers[0]
    assert t1_short.bid_size > t1_short.ask_size


def test_inventory_limit_circuit_breaker():
    """Verify that hitting max inventory triggers EMERGENCY_HALT_ONE_SIDED and halts adding exposure."""
    cfg = AdaptiveMMConfig(max_inventory_limit=5.0)
    optimizer = AdaptiveMarketMakingOptimizer(config=cfg)

    # Max long inventory reached
    optimizer.set_inventory("BTC-USD", 5.0)
    report = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
    )

    assert report.market_regime == ASQuotingRegime.EMERGENCY_HALT_ONE_SIDED
    assert report.bids_allowed is False
    assert report.asks_allowed is True
    # Bid sizes should be zeroed out
    for tier in report.quoting_tiers:
        assert tier.bid_size == 0.0
        assert tier.ask_size > 0.0


def test_volatility_jump_and_drift_regimes():
    """Verify spread widening under volatility jump and reservation shift under trend drift."""
    optimizer = AdaptiveMarketMakingOptimizer()
    optimizer.set_inventory("BTC-USD", 0.0)

    # Normal baseline
    rep_normal = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        has_volatility_jump=False,
    )

    # Jump regime
    rep_jump = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        has_volatility_jump=True,
    )

    assert rep_jump.market_regime == ASQuotingRegime.VOLATILITY_JUMP_WIDE
    assert rep_jump.as_half_spread_bid_bps > rep_normal.as_half_spread_bid_bps
    assert rep_jump.as_half_spread_ask_bps > rep_normal.as_half_spread_ask_bps

    # Strong bullish drift
    rep_drift = optimizer.generate_quote_surface(
        symbol="BTC-USD",
        best_bid=65000.0,
        best_ask=65010.0,
        drift_annualized=1.5,
    )
    assert rep_drift.market_regime == ASQuotingRegime.DRIFT_ACCELERATING
    assert rep_drift.reservation_price > rep_drift.mid_price


def test_fill_tracking_and_probability_decay():
    """Verify execution fill recording updates inventory and Poisson arrival decays monotonically."""
    optimizer = AdaptiveMarketMakingOptimizer()
    optimizer.set_inventory("BTC-USD", 1.0)

    # Record buy fill of 2.5
    new_inv = optimizer.record_fill("BTC-USD", "BUY", 2.5, 65000.0)
    assert new_inv == 3.5
    assert optimizer.get_inventory("BTC-USD") == 3.5

    # Record sell fill of 1.0
    new_inv2 = optimizer.record_fill("BTC-USD", "SELL", 1.0, 65100.0)
    assert new_inv2 == 2.5

    # Test probability decay
    p_tight = optimizer.estimate_fill_probability(2.0)   # 2 bps
    p_wide = optimizer.estimate_fill_probability(20.0)   # 20 bps
    assert 0.0 < p_wide < p_tight <= 1.0
