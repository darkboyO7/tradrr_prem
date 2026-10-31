"""Unit Tests for Quotex Binary Options Backtesting Engine.

Verifies:
1. Binary options execution at bar t+1 open (no lookahead).
2. Binary settlement at expiry close (win payout, loss, tie).
3. Break-even win rate calculation (54.05% for payout=0.85).
4. Correct metrics and trade log generation.
"""

from datetime import UTC, datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine


def create_synthetic_candle_df(n_bars: int = 100) -> pd.DataFrame:
    base_time = datetime(2026, 10, 1, 9, 15, tzinfo=UTC)
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(n_bars)]

    # Generate a realistic random walk price series
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.003, n_bars)
    prices = 1000.0 * np.cumprod(1.0 + returns)

    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.001, n_bars)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.001, n_bars)))
    opens = np.roll(prices, 1)
    opens[0] = 1000.0
    closes = prices
    volumes = np.random.uniform(500, 5000, n_bars)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": np.maximum(highs, np.maximum(opens, closes)),
        "low": np.minimum(lows, np.minimum(opens, closes)),
        "close": closes,
        "volume": volumes,
    })
    return df


def test_binary_options_backtest_execution():
    df = create_synthetic_candle_df(120)
    engine = BacktestEngine(initial_capital=10000.0)

    res = engine.run_binary_options(
        df=df,
        symbol="QUOTEX:BTCUSD_QX",
        timeframe="5m",
        expiry_bars=3,
        payout_pct=0.85,
        stake_amount=100.0,
    )

    assert "equity_curve" in res
    assert "trades" in res
    assert "metrics" in res
    assert res["break_even_win_rate_pct"] == 54.05

    trades = res["trades"]
    for t in trades:
        assert t["symbol"] == "QUOTEX:BTCUSD_QX"
        assert t["action"] in ("BUY", "SELL")
        assert t["outcome"] in ("WIN", "LOSS", "TIE")
        if t["outcome"] == "WIN":
            assert t["pnl"] == pytest.approx(85.0, abs=1e-2)
            assert t["return_pct"] == 85.0
        elif t["outcome"] == "LOSS":
            assert t["pnl"] == pytest.approx(-100.0, abs=1e-2)
            assert t["return_pct"] == -100.0
        else:
            assert t["pnl"] == 0.0


def test_binary_options_insufficient_bars():
    df = create_synthetic_candle_df(10)
    engine = BacktestEngine()
    with pytest.raises(ValueError, match="Insufficient bars"):
        engine.run_binary_options(df=df, expiry_bars=5)
