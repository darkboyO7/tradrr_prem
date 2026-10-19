"""Trading Intelligence Platform - Operational CLI Backtest Runner.

Executes quantitative backtests over historical datasets and displays
institutional risk-adjusted performance metrics.
"""

import argparse

import numpy as np
import pandas as pd

from backtesting.engine import BacktestEngine
from data.historical.parquet_store import ParquetCandleStore


def generate_sample_data(bars: int = 500) -> pd.DataFrame:
    """Generate synthetic market data if local historical store has no partition."""
    np.random.seed(42)
    returns = np.random.normal(0.0004, 0.008, bars)
    # Inject momentum and cyclical trend segments
    returns[50:120] = 0.003
    returns[200:260] = -0.002
    returns[350:420] = 0.004

    closes = 65000.0 * np.exp(np.cumsum(returns))
    highs = closes * 1.004
    lows = closes * 0.996
    opens = closes * 0.999
    volumes = np.random.uniform(50, 500, bars)

    return pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC"),
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


generate_synthetic_candles = generate_sample_data


def run_backtest_cli(symbol: str = "BTC-USD", candles_count: int = 150, initial_capital: float = 100000.0) -> None:
    """Programmatic entry point for running backtests via CLI."""
    df = generate_sample_data(candles_count)
    print("\n========================================================")
    print("  EVENT-DRIVEN BACKTEST EXECUTION SUMMARY")
    print("========================================================")
    print(f"Symbol:           {symbol}")
    print(f"Bars Processed:   {len(df)}")
    print(f"Initial Capital:  ${initial_capital:,.2f}")
    print("========================================================\n")

    engine = BacktestEngine(initial_capital=initial_capital)
    result = engine.run(df, symbol=symbol)
    m = result["metrics"]

    print("---------------- PERFORMANCE REPORT ----------------")
    print(f"Total Cumulative Return:     {m.total_return_pct:+.2f}%")
    print(f"Sharpe Ratio:                 {m.sharpe_ratio:.2f}")
    print(f"Win Rate:                     {m.win_rate:.1f}%")
    print(f"Max Drawdown:                -{m.max_drawdown_pct:.2f}%")
    print("----------------------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Run Trading Intelligence System Backtest")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Market symbol to backtest")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe (e.g. 1h, 15m, 1d)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial portfolio capital")
    args = parser.parse_args()

    store = ParquetCandleStore()
    df = store.read_candles(args.symbol, args.timeframe)
    if df.empty or len(df) < 50:
        print(f"No existing Parquet partition found for {args.symbol} {args.timeframe}. Generating 500 synthetic bars...")
        df = generate_sample_data(500)

    print("\n========================================================")
    print("  TRADING INTELLIGENCE PLATFORM - BACKTEST EXECUTION")
    print("========================================================")
    print(f"Symbol:           {args.symbol}")
    print(f"Timeframe:        {args.timeframe}")
    print(f"Bars Processed:   {len(df)}")
    print(f"Initial Capital:  ${args.capital:,.2f}")
    print("========================================================\n")

    engine = BacktestEngine(initial_capital=args.capital)
    result = engine.run(df, symbol=args.symbol, timeframe=args.timeframe)
    m = result["metrics"]

    print("---------------- PERFORMANCE REPORT ----------------")
    print(f"Total Cumulative Return:     {m.total_return_pct:+.2f}%")
    print(f"Annualized Return (CAGR):     {m.annualized_return_pct:+.2f}%")
    print(f"Sharpe Ratio:                 {m.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:                {m.sortino_ratio:.2f}")
    print(f"Calmar Ratio:                 {m.calmar_ratio:.2f}")
    print(f"Max Drawdown:                -{m.max_drawdown_pct:.2f}% ({m.max_drawdown_bars} bars)")
    print(f"Total Executed Trades:        {m.total_trades}")
    print(f"Win Rate:                     {m.win_rate:.1f}% ({m.winning_trades} wins / {m.losing_trades} losses)")
    print(f"Profit Factor:                {m.profit_factor:.2f}")
    print(f"Payoff Ratio:                 {m.payoff_ratio:.2f}")
    print(f"Expected Return per Trade:   ${m.expectancy_dollars:+,.2f}")
    print(f"Market Exposure Time:         {m.exposure_time_pct:.1f}%")
    print("----------------------------------------------------\n")


if __name__ == "__main__":
    main()
