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
    parser.add_argument("--symbol", type=str, default="NSE:TCS", help="Market symbol to backtest (e.g. NSE:TCS, BSE:RELIANCE, BTC-USD)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (e.g. 5m, 15m, 1h, 1d)")
    parser.add_argument("--capital", type=float, default=500000.0, help="Initial portfolio capital")
    parser.add_argument("--walk-forward", action="store_true", help="Execute chronological walk-forward validation with embargo splits")
    parser.add_argument("--folds", type=int, default=4, help="Number of walk-forward folds (default: 4)")
    args = parser.parse_args()

    df = pd.DataFrame()
    data_source_provenance = "UNKNOWN"

    # Priority 1: Query authentic exchange historical feeds for NSE/BSE symbols
    clean_sym = args.symbol.strip().upper()
    if clean_sym.startswith("NSE:"):
        try:
            from data.ingestion.nse import NSEDataProvider
            nse = NSEDataProvider()
            candles = nse.fetch_historical_candles_sync(clean_sym, timeframe=args.timeframe, limit=240)
            if candles:
                df = pd.DataFrame([c.model_dump() if hasattr(c, "model_dump") else c.__dict__ for c in candles])
                data_source_provenance = f"GROWW_NSE (Real historical feed, {len(candles)} bars)"
        except Exception as exc:
            print(f"Warning: could not fetch live NSE historical feed for {clean_sym}: {exc}")
    elif clean_sym.startswith("BSE:"):
        try:
            from data.ingestion.bse import BSEDataProvider
            bse = BSEDataProvider()
            candles = bse.fetch_historical_candles_sync(clean_sym, timeframe=args.timeframe, limit=240)
            if candles:
                df = pd.DataFrame([c.model_dump() if hasattr(c, "model_dump") else c.__dict__ for c in candles])
                data_source_provenance = f"BSE_FEED (Real historical feed, {len(candles)} bars)"
        except Exception as exc:
            print(f"Warning: could not fetch live BSE historical feed for {clean_sym}: {exc}")

    # Priority 2: Query local Parquet store if not found from live exchange feeds
    if df.empty:
        store = ParquetCandleStore()
        df = store.read_candles(args.symbol, args.timeframe)
        if not df.empty:
            data_source_provenance = f"LOCAL_PARQUET_STORE ({len(df)} bars)"

    # Priority 3: Transparent synthetic benchmark with honest disclosure
    if df.empty or len(df) < 30:
        print(f"Notice: No historical partition found for {args.symbol} {args.timeframe}.")
        print("Generating 300 synthetic benchmark bars for offline algorithm verification (zero live data claim).")
        df = generate_sample_data(300)
        data_source_provenance = "SYNTHETIC_BENCHMARK_SIMULATION"

    # Ensure required columns
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().bfill()

    print("\n========================================================")
    print("  TRADING INTELLIGENCE PLATFORM -- BACKTEST EXECUTION")
    print("========================================================")
    print(f"Symbol:           {args.symbol}")
    print(f"Timeframe:        {args.timeframe}")
    print(f"Bars Processed:   {len(df)}")
    print(f"Data Provenance:  {data_source_provenance}")
    print(f"Initial Capital:  INR {args.capital:,.2f}" if "NSE" in clean_sym or "BSE" in clean_sym else f"Initial Capital:  ${args.capital:,.2f}")
    if "timestamp" in df.columns and len(df) > 0:
        print(f"Date Range:       {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    print("========================================================\n")

    if args.walk_forward:
        from backtesting.walk_forward import WalkForwardValidator
        print("Executing Chronological Walk-Forward Validation (Purged Embargo Splits)...")
        wfv = WalkForwardValidator(n_splits=min(args.folds, max(2, len(df) // 40)), train_ratio=0.60, embargo_bars=3)
        res = wfv.run_walk_forward(df, symbol=args.symbol, timeframe=args.timeframe)
        m = res["oos_metrics"]
        print("\n---------------- WALK-FORWARD OOS PERFORMANCE REPORT ----------------")
        print(f"Walk-Forward Folds:           {res.get('n_folds', 0)}")
        print(f"OOS Cumulative Return:        {m.total_return_pct:+.2f}%")
        print(f"OOS Sharpe Ratio:             {m.sharpe_ratio:.2f}")
        print(f"OOS Max Drawdown:            -{m.max_drawdown_pct:.2f}%")
        print(f"OOS Total Trades:             {m.total_trades}")
        print(f"OOS Win Rate:                 {m.win_rate:.1f}% ({m.winning_trades} wins / {m.losing_trades} losses)")
        print(f"OOS Profit Factor:            {m.profit_factor:.2f}")
        print("--------------------------------------------------------------------\n")
    else:
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
