"""Trading Intelligence Platform - Chronological Walk-Forward Validation Engine.

Implements anchored and rolling Walk-Forward Validation (WFV) with purged embargoes,
generating true out-of-sample performance profiles across regime shifts.
"""

from typing import Any
import pandas as pd
from pydantic import BaseModel

from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetricsCalculator, PerformanceReport


class WalkForwardFold(BaseModel):
    """Container for individual walk-forward fold boundaries and results."""

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: PerformanceReport


class WalkForwardValidator:
    """Orchestrates chronological walk-forward validation without data leakage."""

    def __init__(
        self,
        n_splits: int = 4,
        train_ratio: float = 0.60,
        embargo_bars: int = 5,
        engine: BacktestEngine | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.embargo_bars = embargo_bars
        self.engine = engine or BacktestEngine()

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC-USD",
        timeframe: str = "1h",
    ) -> dict[str, Any]:
        """Execute walk-forward validation and aggregate out-of-sample performance."""
        total_rows = len(df)
        min_required = 150
        if total_rows < min_required:
            raise ValueError(f"Insufficient data for walk-forward validation ({total_rows} < {min_required}).")

        # Calculate fold sizes
        test_chunk_size = int((total_rows * (1.0 - self.train_ratio)) / self.n_splits)
        initial_train_size = int(total_rows * self.train_ratio)

        folds: list[dict[str, Any]] = []
        oos_equity_segments: list[pd.Series] = []
        all_oos_trades: list[dict[str, Any]] = []

        current_cash = self.engine.initial_capital

        for fold_idx in range(self.n_splits):
            train_start = 0  # Expanding window
            train_end = initial_train_size + (fold_idx * test_chunk_size)
            test_start = train_end + self.embargo_bars
            test_end = min(total_rows, test_start + test_chunk_size)

            if test_start >= total_rows or test_start >= test_end:
                break

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            # Run backtest strictly on out-of-sample test_df
            fold_engine = BacktestEngine(
                initial_capital=current_cash,
                limits=self.engine.limits,
                simulator=self.engine.simulator,
            )
            result = fold_engine.run(test_df, symbol=symbol, timeframe=timeframe)

            fold_equity = result["equity_curve"]
            fold_trades = result["trades"]
            fold_metrics = result["metrics"]

            current_cash = float(fold_equity.iloc[-1]) if not fold_equity.empty else current_cash
            oos_equity_segments.append(fold_equity)
            all_oos_trades.extend(fold_trades)

            folds.append({
                "fold": fold_idx + 1,
                "train_bars": len(train_df),
                "test_bars": len(test_df),
                "metrics": fold_metrics,
            })

        # Stitch together out-of-sample equity curve
        if oos_equity_segments:
            combined_oos_equity = pd.concat(oos_equity_segments)
            combined_metrics = PerformanceMetricsCalculator.calculate_metrics(
                equity_curve=combined_oos_equity,
                trades=all_oos_trades,
            )
        else:
            combined_oos_equity = pd.Series()
            combined_metrics = PerformanceReport()

        return {
            "n_folds": len(folds),
            "folds": folds,
            "oos_equity_curve": combined_oos_equity,
            "oos_trades": all_oos_trades,
            "oos_metrics": combined_metrics,
        }
