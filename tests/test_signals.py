"""Tests for Technical Signals and Multi-Source Signal Fusion Engine."""

import pandas as pd

from analysis.regime import MarketRegimeState, TrendRegime, VolatilityRegime, VolumeRegime
from models.base import ModelPrediction, SignalAction
from signals.fusion import SignalFusionEngine
from signals.technical import TechnicalSignal, TechnicalSignalGenerator


def test_technical_signal_generator():
    generator = TechnicalSignalGenerator()

    # Bullish Trend bar
    row_bull = pd.Series({
        "close": 100.0,
        "atr_14": 2.0,
        "rsi_14": 58.0,
        "adx_14": 30.0,
        "plus_di_14": 25.0,
        "minus_di_14": 12.0,
        "ema_bullish_stack": 1,
        "ema_bearish_stack": 0,
        "bb_pct_b_20": 0.65,
        "rvol_20": 1.2,
        "pattern_hammer": 0,
        "pattern_shooting_star": 0,
        "breakout_20": 0,
        "breakdown_20": 0,
    })

    signals = generator.evaluate_bar(row_bull)
    assert len(signals) >= 1
    assert signals[0].setup_name == "TREND_MOMENTUM_BULL"
    assert signals[0].action == SignalAction.BUY

    # Oversold Mean Reversion bar
    row_mr = pd.Series({
        "close": 90.0,
        "atr_14": 2.5,
        "rsi_14": 24.0,
        "adx_14": 15.0,
        "plus_di_14": 10.0,
        "minus_di_14": 15.0,
        "ema_bullish_stack": 0,
        "ema_bearish_stack": 0,
        "bb_pct_b_20": 0.02,
        "rvol_20": 1.0,
        "pattern_hammer": 1,
        "pattern_shooting_star": 0,
        "breakout_20": 0,
        "breakdown_20": 0,
    })

    signals_mr = generator.evaluate_bar(row_mr)
    assert len(signals_mr) >= 1
    assert signals_mr[0].setup_name == "MEAN_REVERSION_OVERSOLD"
    assert signals_mr[0].action == SignalAction.BUY


def test_signal_fusion_conviction_buy():
    engine = SignalFusionEngine()

    regime = MarketRegimeState(
        trend=TrendRegime.STRONG_BULL,
        volatility=VolatilityRegime.NORMAL,
        volume=VolumeRegime.VOLUME_EXPANSION,
        adx=32.0,
        atr_ratio=1.1,
        rvol=1.8,
    )

    tech_signals = [
        TechnicalSignal(
            setup_name="TREND_MOMENTUM_BULL",
            action=SignalAction.BUY,
            strength=0.85,
            reasons=["Bullish trend alignment"],
        )
    ]

    ml_prediction = ModelPrediction(
        action=SignalAction.BUY,
        probability=0.72,
        expected_return=0.018,
        horizon_bars=5,
    )

    pattern_consensus = {
        "sample_size": 10,
        "direction": "BULLISH",
        "confidence": 0.70,
        "win_rate": 0.80,
    }

    decision = engine.fuse(
        symbol="BTC-USD",
        current_price=65000.0,
        atr=500.0,
        regime=regime,
        technical_signals=tech_signals,
        ml_prediction=ml_prediction,
        pattern_consensus=pattern_consensus,
    )

    assert decision.action == SignalAction.BUY
    assert decision.confidence >= 0.70
    assert decision.stop_loss == 65000.0 - (1.5 * 500.0)
    assert decision.take_profit == 65000.0 + (3.0 * 500.0)
    assert decision.risk_reward_ratio == 2.0
    assert len(decision.reasons) > 0
    assert "CONVICTION BUY" in decision.reasons[0]


def test_signal_fusion_no_trade_on_conflicts():
    engine = SignalFusionEngine()

    regime = MarketRegimeState(
        trend=TrendRegime.RANGING,
        volatility=VolatilityRegime.NORMAL,
        volume=VolumeRegime.NORMAL_VOLUME,
    )

    # Empty/weak signals
    decision = engine.fuse(
        symbol="ETH-USD",
        current_price=3000.0,
        atr=50.0,
        regime=regime,
        technical_signals=[],
        ml_prediction=ModelPrediction(action=SignalAction.NO_TRADE, probability=0.40),
    )

    assert decision.action == SignalAction.NO_TRADE
    assert "NO-TRADE" in decision.reasons[0]
