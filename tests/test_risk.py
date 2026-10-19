"""Tests for Position Sizing and Pre-Trade Risk Engine."""

from models.base import SignalAction
from risk.engine import RiskDecisionStatus, RiskEngine
from risk.sizing import PositionSizer
from signals.fusion import FinalSignalDecision


def test_fixed_fractional_position_sizer():
    # Equity = $100,000. Risk 1.5% = $1,500.
    # Entry = $100, Stop Loss = $95 (Risk per unit = $5).
    # Size = 1500 / 5 = 300 units ($30,000 notional value, <= 25% max position = $25,000 cap).
    # With 25% max position cap on $100k equity = $25,000 / $100 = 250 units max.
    size = PositionSizer.calculate_fixed_fractional_size(
        equity=100000.0,
        entry_price=100.0,
        stop_loss=95.0,
        risk_per_trade_pct=0.015,
        max_position_pct=0.25,
    )
    assert size == 250.0  # Capped by 25% max notional

    # Without cap binding:
    # Entry = 100, SL = 90 ($10 risk per unit). Raw size = 1500 / 10 = 150 units ($15,000 notional < $25,000).
    size_uncapped = PositionSizer.calculate_fixed_fractional_size(
        equity=100000.0,
        entry_price=100.0,
        stop_loss=90.0,
        risk_per_trade_pct=0.015,
        max_position_pct=0.25,
    )
    assert size_uncapped == 150.0


def test_fractional_kelly_sizer():
    # Win rate 60%, Payoff ratio 2.0
    # Full Kelly: (0.6 * 2 - 0.4) / 2 = (1.2 - 0.4)/2 = 0.40 (40%)
    # Quarter Kelly: 0.40 * 0.25 = 0.10 -> Capped at hard ceiling 0.05
    k = PositionSizer.calculate_fractional_kelly(win_rate=0.60, payoff_ratio=2.0, fraction=0.25)
    assert k == 0.05

    # Negative expectancy: Win rate 30%, Payoff 1.0 -> 0.0
    k_neg = PositionSizer.calculate_fractional_kelly(win_rate=0.30, payoff_ratio=1.0)
    assert k_neg == 0.0


def test_risk_engine_trade_approval():
    engine = RiskEngine(initial_capital=100000.0)

    signal = FinalSignalDecision(
        symbol="BTC-USD",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=60000.0,
        stop_loss=58500.0,  # 1500 distance
        take_profit=63000.0,  # 3000 distance
        risk_reward_ratio=2.0,
    )

    decision = engine.evaluate_trade(signal)
    assert decision.status == RiskDecisionStatus.APPROVED
    assert decision.approved_quantity > 0.0
    assert decision.approved_quantity == decision.requested_quantity


def test_risk_engine_trade_resizing():
    engine = RiskEngine(initial_capital=100000.0)

    signal = FinalSignalDecision(
        symbol="ETH-USD",
        action=SignalAction.BUY,
        confidence=0.80,
        entry_price=1000.0,
        stop_loss=900.0,  # 100 distance (1.5% risk = $1500 -> 15 units allowed, notional $15k <= $25k cap)
        take_profit=1300.0,
        risk_reward_ratio=3.0,
    )

    # Request oversized 50 units (exceeds 15 units safe size)
    decision = engine.evaluate_trade(signal, requested_quantity=50.0)
    assert decision.status == RiskDecisionStatus.RESIZED
    assert decision.requested_quantity == 50.0
    assert decision.approved_quantity == 15.0


def test_risk_engine_circuit_breakers():
    engine = RiskEngine(initial_capital=100000.0)

    # 1. Simulate a severe 15% drawdown (limit is 12%)
    engine.update_portfolio_state(current_equity=85000.0, open_positions={})

    signal = FinalSignalDecision(
        symbol="BTC-USD",
        action=SignalAction.BUY,
        confidence=0.90,
        entry_price=60000.0,
        stop_loss=59000.0,
        take_profit=62000.0,
        risk_reward_ratio=2.0,
    )

    decision = engine.evaluate_trade(signal)
    assert decision.status == RiskDecisionStatus.REJECTED
    assert "circuit breaker" in decision.rejection_reason.lower()


def test_risk_engine_rejects_low_risk_reward():
    engine = RiskEngine(initial_capital=100000.0)

    signal = FinalSignalDecision(
        symbol="SOL-USD",
        action=SignalAction.BUY,
        confidence=0.75,
        entry_price=150.0,
        stop_loss=140.0,  # 10 distance
        take_profit=160.0,  # 10 distance
        risk_reward_ratio=1.0,  # Below 1.5 minimum limit
    )

    decision = engine.evaluate_trade(signal)
    assert decision.status == RiskDecisionStatus.REJECTED
    assert "risk/reward" in decision.rejection_reason.lower()
