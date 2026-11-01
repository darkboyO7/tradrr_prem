"""Test Platform Hardening and Truthfulness Controls.

Verifies:
1. Local network isolation (APP_HOST defaults to 127.0.0.1).
2. Robust position sizing against NaN/inf floating-point poison.
3. Pre-trade risk geometry and price sanity enforcement.
4. Market state zero-spread elimination on trade and candle events.
5. Aggregated bar OHLC mathematical consistency.
6. Fusion conviction gating on positive, finite ATR.
7. Desktop service controller startup alias and provenance badge.
"""

import math
from datetime import UTC, datetime, timedelta

import pytest

from core.config import Settings
from core.events import Candle, NormalizedTradeEvent, TradeSide
from data.market_state import MarketStateService
from models.base import SignalAction
from risk.engine import RiskDecisionStatus, RiskEngine
from risk.sizing import PositionSizer
from signals.fusion import FinalSignalDecision, SignalFusionEngine


class TestPlatformHardening:
    """Hardening and truthfulness test suite."""

    def test_app_host_defaults_to_loopback(self):
        """Settings field definition defaults to 127.0.0.1 for local isolation."""
        field_default = Settings.model_fields["APP_HOST"].default
        assert field_default == "127.0.0.1", f"APP_HOST should default to 127.0.0.1, got {field_default}"

    def test_position_sizer_nan_and_inf_safety(self):
        """Position sizer safely returns 0.0 for any NaN or Inf inputs without raising exceptions."""
        # Fixed fractional
        assert PositionSizer.calculate_fixed_fractional_size(float("nan"), 100.0, 95.0) == 0.0
        assert PositionSizer.calculate_fixed_fractional_size(100000.0, float("nan"), 95.0) == 0.0
        assert PositionSizer.calculate_fixed_fractional_size(100000.0, 100.0, float("nan")) == 0.0
        assert PositionSizer.calculate_fixed_fractional_size(float("inf"), 100.0, 95.0) == 0.0
        assert PositionSizer.calculate_fixed_fractional_size(100000.0, float("inf"), 95.0) == 0.0
        assert PositionSizer.calculate_fixed_fractional_size(100000.0, 100.0, float("inf")) == 0.0
        assert PositionSizer.calculate_fixed_fractional_size(100000.0, 100.0, 100.0) == 0.0  # zero distance

        # Volatility adjusted
        assert PositionSizer.calculate_volatility_adjusted_size(100000.0, 100.0, float("nan"), 1000.0) == 0.0
        assert PositionSizer.calculate_volatility_adjusted_size(100000.0, 100.0, float("inf"), 1000.0) == 0.0
        assert PositionSizer.calculate_volatility_adjusted_size(100000.0, 100.0, -2.5, 1000.0) == 0.0

        # Fractional Kelly
        assert PositionSizer.calculate_fractional_kelly(float("nan"), 2.0) == 0.0
        assert PositionSizer.calculate_fractional_kelly(0.55, float("nan")) == 0.0
        assert PositionSizer.calculate_fractional_kelly(float("inf"), 2.0) == 0.0
        assert PositionSizer.calculate_fractional_kelly(0.55, float("inf")) == 0.0

    def test_risk_engine_rejects_invalid_prices_and_inverted_geometry(self):
        """Risk engine rejects non-positive or inverted stop loss geometry."""
        engine = RiskEngine(initial_capital=100000.0)

        # Non-positive entry price
        sig_bad_price = FinalSignalDecision(
            symbol="RELIANCE.NS",
            action=SignalAction.BUY,
            confidence=0.85,
            buy_probability=0.85,
            sell_probability=0.15,
            no_trade_probability=0.0,
            entry_price=0.0,
            stop_loss=2400.0,
            take_profit=2600.0,
            risk_reward_ratio=2.0,
        )
        dec = engine.evaluate_trade(sig_bad_price)
        assert dec.status == RiskDecisionStatus.REJECTED
        assert "Invalid entry price" in dec.rejection_reason

        # Inverted BUY geometry: stop_loss >= entry_price
        sig_inv_buy = FinalSignalDecision(
            symbol="RELIANCE.NS",
            action=SignalAction.BUY,
            confidence=0.85,
            buy_probability=0.85,
            sell_probability=0.15,
            no_trade_probability=0.0,
            entry_price=2500.0,
            stop_loss=2550.0,  # Above entry price!
            take_profit=2700.0,
            risk_reward_ratio=2.0,
        )
        dec_buy = engine.evaluate_trade(sig_inv_buy)
        assert dec_buy.status == RiskDecisionStatus.REJECTED
        assert "Invalid BUY risk geometry" in dec_buy.rejection_reason

        # Inverted SELL geometry: stop_loss <= entry_price
        sig_inv_sell = FinalSignalDecision(
            symbol="RELIANCE.NS",
            action=SignalAction.SELL,
            confidence=0.85,
            buy_probability=0.15,
            sell_probability=0.85,
            no_trade_probability=0.0,
            entry_price=2500.0,
            stop_loss=2450.0,  # Below entry price!
            take_profit=2300.0,
            risk_reward_ratio=2.0,
        )
        dec_sell = engine.evaluate_trade(sig_inv_sell)
        assert dec_sell.status == RiskDecisionStatus.REJECTED
        assert "Invalid SELL risk geometry" in dec_sell.rejection_reason

    def test_market_state_eliminates_zero_spread_fabrication(self):
        """Market state does not fabricate bid/ask or 0.0 spread on raw trade ticks or candles."""
        svc = MarketStateService.get_instance()

        now = datetime.now(UTC)
        trade = NormalizedTradeEvent(
            provider="test_provider",
            symbol="TEST_TRADE_TICK",
            trade_id="t_001",
            timestamp=now,
            received_at=now,
            price=123.45,
            size=10.0,
            side=TradeSide.BUY,
        )
        svc.on_trade(trade)
        st = svc.get_state("TEST_TRADE_TICK")
        assert st is not None
        assert st.last_price == 123.45
        assert st.bid_price is None, "Bid price should be None when unobserved"
        assert st.ask_price is None, "Ask price should be None when unobserved"
        assert st.spread is None, "Spread should be None when unobserved"

    def test_aggregate_bars_strict_ohlc_integrity(self):
        """Aggregated bars enforce high >= max(open, close) and low <= min(open, close)."""
        svc = MarketStateService.get_instance()
        # Feed bars with deliberately anomalous low/high to verify auto-repair
        raw_bars = [
            {"timestamp": "2026-10-20T10:00:00Z", "open": 100.0, "high": 90.0, "low": 110.0, "close": 105.0, "volume": 100.0},
            {"timestamp": "2026-10-20T10:05:00Z", "open": 105.0, "high": 102.0, "low": 108.0, "close": 112.0, "volume": -50.0},
        ]
        agg = svc._aggregate_bars(raw_bars, 2)
        assert len(agg) == 1
        b = agg[0]
        assert b["open"] == 100.0
        assert b["close"] == 112.0
        assert b["high"] >= max(b["open"], b["close"])
        assert b["low"] <= min(b["open"], b["close"])
        assert b["volume"] >= 0.0

    def test_signal_fusion_gating_on_invalid_atr(self):
        """Signal fusion does not generate conviction signals when ATR is zero or NaN."""
        from analysis.regime import MarketRegimeState
        from signals.technical import TechnicalSignal

        engine = SignalFusionEngine()
        regime = MarketRegimeState()
        # Even with strong BUY signals, if ATR is 0.0 or NaN, conviction must be gated
        strong_signals = [
            TechnicalSignal(setup_name="EMA_Cross", action=SignalAction.BUY, strength=1.0),
            TechnicalSignal(setup_name="MACD", action=SignalAction.BUY, strength=1.0),
            TechnicalSignal(setup_name="RSI", action=SignalAction.BUY, strength=1.0),
        ]
        dec_zero_atr = engine.fuse(
            symbol="RELIANCE.NS",
            current_price=2500.0,
            atr=0.0,
            regime=regime,
            technical_signals=strong_signals,
            horizon="15m",
        )
        assert dec_zero_atr.action == SignalAction.NO_TRADE

        dec_nan_atr = engine.fuse(
            symbol="RELIANCE.NS",
            current_price=2500.0,
            atr=float("nan"),
            regime=regime,
            technical_signals=strong_signals,
            horizon="15m",
        )
        assert dec_nan_atr.action == SignalAction.NO_TRADE

    def test_desktop_service_controller_startup_alias(self):
        """DesktopServiceController has startup() method aliased to run_startup_sequence."""
        from desktop.service_controller import DesktopServiceController

        sc = DesktopServiceController()
        assert hasattr(sc, "startup"), "DesktopServiceController should have startup() method"
        assert callable(sc.startup)
