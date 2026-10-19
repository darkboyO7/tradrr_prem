"""Tests for Backtesting Engine and Quantitative Performance Analytics."""

import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetricsCalculator, PerformanceReport


def test_performance_metrics_calculator():
    # 100 days of equity starting at $100,000, ending at $120,000
    equity_vals = [100000.0]
    for i in range(1, 100):
        equity_vals.append(equity_vals[-1] * (1.0 + (0.002 if i % 2 == 0 else -0.0005)))
    equity_curve = pd.Series(equity_vals)

    trades = [
        {"pnl": 500.0, "return_pct": 2.5, "duration_bars": 5},
        {"pnl": -200.0, "return_pct": -1.0, "duration_bars": 3},
        {"pnl": 800.0, "return_pct": 4.0, "duration_bars": 8},
        {"pnl": -150.0, "return_pct": -0.8, "duration_bars": 2},
    ]

    report = PerformanceMetricsCalculator.calculate_metrics(equity_curve, trades, periods_per_year=252)
    assert report.total_return_pct > 0
    assert report.sharpe_ratio > 0
    assert report.max_drawdown_pct >= 0.0
    assert report.total_trades == 4
    assert report.winning_trades == 2
    assert report.losing_trades == 2
    assert report.win_rate == 50.0
    assert report.profit_factor == pytest.approx(1300.0 / 350.0, rel=1e-2)
    assert report.expectancy_dollars > 0


def test_backtest_engine_run():
    np.random.seed(42)
    bars = 100
    returns = np.random.normal(0.0008, 0.008, bars)
    # Inject a strong bull trend section
    returns[20:50] = 0.005
    closes = 50000.0 * np.exp(np.cumsum(returns))
    highs = closes * 1.005
    lows = closes * 0.995
    opens = closes * 0.999
    volumes = np.random.uniform(50, 200, bars)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    engine = BacktestEngine(initial_capital=100000.0)
    result = engine.run(df, symbol="BTC-USD", timeframe="1h")

    assert "equity_curve" in result
    assert "trades" in result
    assert "metrics" in result

    equity = result["equity_curve"]
    assert len(equity) == bars
    assert equity.iloc[0] == 100000.0

    metrics: PerformanceReport = result["metrics"]
    assert isinstance(metrics.sharpe_ratio, float)
    assert isinstance(metrics.max_drawdown_pct, float)
