"""Tests for Cross-Venue Synthetic Funding Rate Arbitrage & Basis Trading Engine."""

import math
import pytest

from analysis.basis_arbitrage import (
    BasisArbitrageScanner,
    BasisOpportunity,
    BasisPosition,
    BasisPositionTracker,
    BasisTradeType,
    PerpetualFundingRate,
    SpotQuote,
)


def test_funding_rate_annualization():
    """Verify funding rate annualization across 8-hour and 1-hour intervals."""
    # 0.01% (0.0001) per 8h = 3 * 365 * 0.01% = 10.95% APR
    rate_8h = PerpetualFundingRate(
        venue="binance",
        symbol="BTCUSDT",
        rate_per_interval=0.0001,
        interval_hours=8,
        mark_price=65000.0,
        index_price=65000.0,
    )
    assert math.isclose(rate_8h.annualized_rate_pct, 10.95, rel_tol=1e-3)

    # 0.005% per 1h = 24 * 365 * 0.005% = 43.8% APR
    rate_1h = PerpetualFundingRate(
        venue="dydx",
        symbol="BTC-USD",
        rate_per_interval=0.00005,
        interval_hours=1,
        mark_price=65000.0,
        index_price=65000.0,
    )
    assert math.isclose(rate_1h.annualized_rate_pct, 43.80, rel_tol=1e-3)


def test_cash_and_carry_scanning():
    """Verify positive basis and positive funding rates generate cash-and-carry opportunities."""
    scanner = BasisArbitrageScanner(default_taker_fee_bps=4.0, default_maker_fee_bps=1.0)

    # Spot on Coinbase: Ask $65,000
    scanner.register_spot_quote("BTC", "coinbase", "BTC-USD", bid=64990.0, ask=65000.0)

    # Perp on Binance: Mark $65,300 (+46.15 bps basis), funding 0.03% per 8h (~32.85% APR)
    scanner.register_perp_rate(
        "BTC", "binance", "BTCUSDT", rate_per_interval=0.0003, mark_price=65300.0, index_price=65000.0
    )

    opps = scanner.scan_opportunities(min_net_apr_pct=5.0)
    assert len(opps) >= 1

    opp = opps[0]
    assert opp.strategy_type == BasisTradeType.CASH_AND_CARRY
    assert opp.underlying_asset == "BTC"
    assert opp.long_leg_venue == "coinbase"
    assert opp.short_leg_venue == "binance"
    assert opp.basis_spread_bps > 40.0
    assert opp.net_funding_apr_pct > 30.0
    assert opp.estimated_net_apr_pct > 30.0
    assert opp.recommended_leverage == 2.0


def test_reverse_cash_and_carry_scanning():
    """Verify deeply negative funding rates generate reverse cash-and-carry opportunities."""
    scanner = BasisArbitrageScanner(default_borrow_rate_apr=2.5)

    # Spot on Kraken: Bid $65,100
    scanner.register_spot_quote("BTC", "kraken", "BTC-USD", bid=65100.0, ask=65110.0)

    # Perp on Bybit: Discount mark $64,800 (-46 bps), negative funding -0.04% per 8h (longs get paid)
    scanner.register_perp_rate(
        "BTC", "bybit", "BTCUSD", rate_per_interval=-0.0004, mark_price=64800.0, index_price=65100.0
    )

    opps = scanner.scan_opportunities(min_net_apr_pct=5.0)
    reverse_opps = [o for o in opps if o.strategy_type == BasisTradeType.REVERSE_CASH_AND_CARRY]
    assert len(reverse_opps) >= 1

    opp = reverse_opps[0]
    assert opp.long_leg_venue == "bybit"  # Long perp
    assert opp.short_leg_venue == "kraken"  # Short spot
    assert opp.spot_borrow_apr_pct == 2.5
    assert opp.net_funding_apr_pct > 40.0  # -(-43.8%) = +43.8%
    assert opp.estimated_net_apr_pct > 35.0


def test_perp_perp_funding_arbitrage_scanning():
    """Verify cross-venue funding rate divergence creates delta-neutral perp-perp carry trade."""
    scanner = BasisArbitrageScanner()

    # Venue A: Binance low funding rate 0.01% (10.95% APR)
    scanner.register_perp_rate(
        "SOL", "binance", "SOLUSDT", rate_per_interval=0.0001, mark_price=150.0, index_price=150.0
    )

    # Venue B: Hyperliquid high funding rate 0.06% (65.70% APR)
    scanner.register_perp_rate(
        "SOL", "hyperliquid", "SOL-PERP", rate_per_interval=0.0006, mark_price=150.2, index_price=150.0
    )

    opps = scanner.scan_opportunities(min_net_apr_pct=10.0)
    perp_arb = [o for o in opps if o.strategy_type == BasisTradeType.PERP_PERP_FUNDING_ARB]
    assert len(perp_arb) >= 1

    opp = perp_arb[0]
    assert opp.long_leg_venue == "binance"
    assert opp.short_leg_venue == "hyperliquid"
    assert opp.net_funding_apr_pct > 50.0
    assert opp.recommended_leverage == 3.0


def test_basis_position_tracker():
    """Verify position opening, funding payments, and convergence mark tracking."""
    tracker = BasisPositionTracker()

    sample_opp = BasisOpportunity(
        strategy_type=BasisTradeType.CASH_AND_CARRY,
        underlying_asset="BTC",
        long_leg_venue="coinbase",
        long_leg_symbol="BTC-USD",
        long_leg_price=65000.0,
        short_leg_venue="binance",
        short_leg_symbol="BTCUSDT",
        short_leg_price=65200.0,
        basis_spread_bps=30.77,
        annualized_basis_yield_pct=37.4,
        net_funding_apr_pct=21.9,
        spot_borrow_apr_pct=0.0,
        trading_fee_bps=10.0,
        estimated_net_apr_pct=50.0,
        recommended_leverage=2.0,
        liquidation_buffer_pct=30.0,
    )

    pos = tracker.open_position(opportunity=sample_opp, quantity=2.0)
    assert pos.quantity == 2.0
    assert pos.total_capital_deployed_usd == 2.0 * (65000.0 + 65200.0)

    # Record funding payment of $50
    updated_pos = tracker.accrue_funding_payment(position_id=pos.position_id, funding_cashflow_usd=50.0)
    assert updated_pos is not None
    assert updated_pos.cumulative_funding_collected_usd == 50.0
    assert updated_pos.realized_funding_yield_pct > 0.0

    # Mark update: Spot rises to 66,000 (+1,000), Perp rises to 66,100 (+900) -> Basis compressed by $100
    pos_mtm = tracker.update_mark(
        position_id=pos.position_id,
        current_long_price=66000.0,
        current_short_price=66100.0,
    )
    assert pos_mtm is not None
    # Long PnL = 2 * (66000 - 65000) = +2000
    # Short PnL = 2 * (65200 - 66100) = -1800
    # Total unrealized PnL = +200
    assert pos_mtm.unrealized_pnl_usd == 200.0
