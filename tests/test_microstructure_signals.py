"""Tests for Microstructural Alpha Signals and Fusion Engine Integration."""

import pandas as pd
import pytest

from analysis.regime import MarketRegimeState, TrendRegime, VolatilityRegime
from models.base import SignalAction
from signals.fusion import FinalSignalDecision, SignalFusionEngine
from signals.microstructure import MicrostructureSignal, MicrostructureSignalGenerator
from signals.technical import TechnicalSignal


def test_microstructure_signal_generator_imbalance():
    """Verify OBI generates directional BUY and SELL signals."""
    generator = MicrostructureSignalGenerator(imbalance_threshold=0.35)

    # 1. Bullish imbalance
    row_buy = pd.Series({"obi_top5": 0.45, "spread_bps": 6.0, "liquidity_vacuum": 0})
    sig_buy = generator.evaluate_bar(row_buy)
    assert sig_buy is not None
    assert sig_buy.action == SignalAction.BUY
    assert sig_buy.signal_name == "ORDER_BOOK_IMBALANCE_BUY"
    assert sig_buy.confidence > 0.60

    # 2. Bearish imbalance
    row_sell = pd.Series({"obi_top5": -0.50, "spread_bps": 5.5, "liquidity_vacuum": 0})
    sig_sell = generator.evaluate_bar(row_sell)
    assert sig_sell is not None
    assert sig_sell.action == SignalAction.SELL
    assert sig_sell.signal_name == "ORDER_BOOK_IMBALANCE_SELL"

    # 3. Neutral state (within threshold)
    row_neutral = pd.Series({"obi_top5": 0.10, "spread_bps": 5.0, "liquidity_vacuum": 0})
    sig_neutral = generator.evaluate_bar(row_neutral)
    assert sig_neutral is None


def test_microstructure_liquidity_vacuum_circuit_breaker():
    """Verify spread expansion triggers LIQUIDITY_VACUUM_HALT."""
    generator = MicrostructureSignalGenerator(max_acceptable_spread_bps=20.0)
    row_vacuum = pd.Series({"obi_top5": 0.60, "spread_bps": 40.0, "liquidity_vacuum": 1})

    sig = generator.evaluate_bar(row_vacuum)
    assert sig is not None
    assert sig.action == SignalAction.NO_TRADE
    assert sig.signal_name == "LIQUIDITY_VACUUM_HALT"
    assert sig.is_vacuum is True


def test_signal_fusion_with_microstructure():
    """Verify SignalFusionEngine handles vacuum halt and OBI confirmation."""
    fusion = SignalFusionEngine(min_confidence_threshold=0.40)
    regime = MarketRegimeState()

    tech_buy = [
        TechnicalSignal(
            setup_name="TREND_MOMENTUM_BULL",
            action=SignalAction.BUY,
            strength=0.75,
            horizon_bars=10,
        )
    ]

    # Case A: Vacuum halts trade despite technical buy setup
    vacuum_sig = MicrostructureSignal(
        signal_name="LIQUIDITY_VACUUM_HALT",
        action=SignalAction.NO_TRADE,
        confidence=0.0,
        imbalance=0.0,
        spread_bps=50.0,
        is_vacuum=True,
    )

    decision_halt = fusion.fuse(
        symbol="BTC-USD",
        current_price=65000.0,
        atr=500.0,
        regime=regime,
        technical_signals=tech_buy,
        microstructure_signal=vacuum_sig,
    )
    assert decision_halt.action == SignalAction.NO_TRADE
    assert "vacuum" in decision_halt.reasons[0].lower()

    # Case B: Order book imbalance confirms technical buy setup
    obi_buy_sig = MicrostructureSignal(
        signal_name="ORDER_BOOK_IMBALANCE_BUY",
        action=SignalAction.BUY,
        confidence=0.80,
        imbalance=0.55,
        spread_bps=5.0,
        is_vacuum=False,
        reasons=["Strong bid depth imbalance"],
    )

    decision_confirm = fusion.fuse(
        symbol="BTC-USD",
        current_price=65000.0,
        atr=500.0,
        regime=regime,
        technical_signals=tech_buy,
        microstructure_signal=obi_buy_sig,
    )
    assert decision_confirm.action == SignalAction.BUY
    assert any("bid depth" in r.lower() for r in decision_confirm.reasons)
