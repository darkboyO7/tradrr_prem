"""Trading Intelligence Platform - Quantitative Performance & Risk Analytics.

Calculates institutional-grade backtesting and paper trading metrics:
Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor, Expectancy, and Payoff Ratio.
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class PerformanceReport(BaseModel):
    """Exhaustive quantitative performance and risk metrics report."""

    total_return_pct: float = Field(default=0.0, description="Total net cumulative return percentage")
    annualized_return_pct: float = Field(default=0.0, description="Compound annual growth rate (CAGR)")
    sharpe_ratio: float = Field(default=0.0, description="Annualized risk-adjusted excess return")
    sortino_ratio: float = Field(default=0.0, description="Downside risk-adjusted excess return")
    calmar_ratio: float = Field(default=0.0, description="Annualized return divided by maximum drawdown")
    max_drawdown_pct: float = Field(default=0.0, description="Peak-to-trough maximum portfolio decline")
    max_drawdown_bars: int = Field(default=0, description="Maximum duration of drawdown in bars")
    total_trades: int = Field(default=0, description="Total completed trade executions")
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = Field(default=0.0, description="Percentage of profitable trades")
    loss_rate: float = Field(default=0.0, description="Percentage of unprofitable trades")
    profit_factor: float = Field(default=0.0, description="Gross profits divided by gross losses")
    payoff_ratio: float = Field(default=0.0, description="Average win divided by average loss magnitude")
    expectancy_dollars: float = Field(default=0.0, description="Mathematical dollar expectancy per trade")
    avg_trade_return_pct: float = 0.0
    exposure_time_pct: float = Field(default=0.0, description="Percentage of bars with active open market exposure")


class PerformanceMetricsCalculator:
    """Computes comprehensive portfolio and trade-level performance statistics."""

    @staticmethod
    def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
        """Calculate percentage drawdown from rolling historical peak equity."""
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        return drawdowns

    @classmethod
    def calculate_metrics(
        cls,
        equity_curve: pd.Series,
        trades: list[dict[str, Any]],
        periods_per_year: int = 252 * 24 * 60,  # Default for 1-minute bars in 24/7 crypto
        risk_free_rate: float = 0.0,
    ) -> PerformanceReport:
        """Compute full suite of statistical and financial performance metrics.

        Args:
            equity_curve: Chronological Series of portfolio equity values over time.
            trades: List of trade dictionaries containing 'pnl', 'return_pct', 'duration_bars'.
            periods_per_year: Frequency scaling factor for annualization.
            risk_free_rate: Benchmark risk-free interest rate.
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return PerformanceReport()

        initial_equity = float(equity_curve.iloc[0])
        final_equity = float(equity_curve.iloc[-1])
        total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100.0

        # Returns series
        bar_returns = equity_curve.pct_change().dropna()
        n_bars = len(bar_returns)

        # CAGR
        years = max(n_bars / periods_per_year, 1.0 / periods_per_year)
        if final_equity > 0 and initial_equity > 0:
            cagr = ((final_equity / initial_equity) ** (1.0 / years) - 1.0) * 100.0
        else:
            cagr = -100.0

        # Drawdown analysis
        dd_series = cls.calculate_drawdown_series(equity_curve)
        max_dd_pct = float(abs(dd_series.min())) * 100.0

        # Drawdown duration in bars
        is_in_dd = dd_series < 0
        dd_duration = 0
        max_dd_duration = 0
        for in_dd in is_in_dd:
            if in_dd:
                dd_duration += 1
                if dd_duration > max_dd_duration:
                    max_dd_duration = dd_duration
            else:
                dd_duration = 0

        # Sharpe ratio
        mean_ret = float(bar_returns.mean())
        std_ret = float(bar_returns.std(ddof=1)) if len(bar_returns) > 1 else 0.0
        if std_ret > 0:
            sharpe = (mean_ret - (risk_free_rate / periods_per_year)) / std_ret * np.sqrt(periods_per_year)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = bar_returns[bar_returns < 0]
        downside_std = float(downside_returns.std(ddof=1)) if len(downside_returns) > 1 else 0.0
        if downside_std > 0:
            sortino = (mean_ret - (risk_free_rate / periods_per_year)) / downside_std * np.sqrt(periods_per_year)
        else:
            sortino = sharpe if sharpe > 0 else 0.0

        # Calmar ratio
        calmar = (cagr / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Trade-level statistics
        n_trades = len(trades)
        if n_trades > 0:
            pnls = np.array([t.get("pnl", 0.0) for t in trades])
            ret_pcts = np.array([t.get("return_pct", 0.0) for t in trades])

            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]

            n_wins = len(wins)
            n_losses = len(losses)
            win_rate = (n_wins / n_trades) * 100.0
            loss_rate = (n_losses / n_trades) * 100.0

            sum_win = float(np.sum(wins)) if n_wins > 0 else 0.0
            sum_loss = float(abs(np.sum(losses))) if n_losses > 0 else 0.0
            profit_factor = (sum_win / sum_loss) if sum_loss > 0 else (99.0 if sum_win > 0 else 0.0)

            mean_win = float(np.mean(wins)) if n_wins > 0 else 0.0
            mean_loss = float(abs(np.mean(losses))) if n_losses > 0 else 0.0
            payoff_ratio = (mean_win / mean_loss) if mean_loss > 0 else 0.0

            # Expectancy: (WinRate * AvgWin) - (LossRate * AvgLoss)
            p_win = n_wins / n_trades
            p_loss = n_losses / n_trades
            expectancy = (p_win * mean_win) - (p_loss * mean_loss)
            avg_trade_ret = float(np.mean(ret_pcts))

            total_trade_bars = sum(t.get("duration_bars", 1) for t in trades)
            exposure_time_pct = min(100.0, (total_trade_bars / n_bars) * 100.0) if n_bars > 0 else 0.0
        else:
            n_wins = 0
            n_losses = 0
            win_rate = 0.0
            loss_rate = 0.0
            profit_factor = 0.0
            payoff_ratio = 0.0
            expectancy = 0.0
            avg_trade_ret = 0.0
            exposure_time_pct = 0.0

        return PerformanceReport(
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=round(cagr, 2),
            sharpe_ratio=round(float(sharpe), 2),
            sortino_ratio=round(float(sortino), 2),
            calmar_ratio=round(float(calmar), 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_bars=max_dd_duration,
            total_trades=n_trades,
            winning_trades=n_wins,
            losing_trades=n_losses,
            win_rate=round(win_rate, 2),
            loss_rate=round(loss_rate, 2),
            profit_factor=round(profit_factor, 2),
            payoff_ratio=round(payoff_ratio, 2),
            expectancy_dollars=round(expectancy, 2),
            avg_trade_return_pct=round(avg_trade_ret, 3),
            exposure_time_pct=round(exposure_time_pct, 2),
        )
