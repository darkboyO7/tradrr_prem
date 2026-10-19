"""Trading Intelligence Platform - Statistical Profiling Engine.

Calculates rigorous statistical expectancy, payoff ratios, win rates, skewness,
kurtosis, and profit factors for conditioned market regimes, signals, and setups.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class StatisticalProfile(BaseModel):
    """Rigorous statistical metrics for conditional return distributions."""

    condition_name: str
    horizon_bars: int
    sample_size: int = 0
    win_rate: float = Field(default=0.0, description="Proportion of positive forward outcomes")
    loss_rate: float = Field(default=0.0, description="Proportion of negative forward outcomes")
    mean_return: float = 0.0
    median_return: float = 0.0
    mean_win: float = 0.0
    mean_loss: float = 0.0
    payoff_ratio: float = Field(default=0.0, description="Mean win divided by mean loss magnitude")
    expectancy: float = Field(default=0.0, description="(WinRate * MeanWin) - (LossRate * Abs(MeanLoss))")
    profit_factor: float = Field(default=0.0, description="Sum of positive returns / sum of negative returns")
    std_dev: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    t_stat: float = Field(default=0.0, description="T-statistic for H0: mean return == 0")


class StatisticalProfiler:
    """Computes conditional forward return distributions for any trading setup, pattern, or regime."""

    @staticmethod
    def calculate_forward_returns(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
        """Append forward returns for specified bar horizons (e.g. forward 1, 3, 5, 10 bars)."""
        if horizons is None:
            horizons = [1, 3, 5, 10, 20]

        res = df.copy()
        c = res["close"]
        for h in horizons:
            # fwd_return_h = (close[t+h] - close[t]) / close[t]
            res[f"fwd_return_{h}"] = (c.shift(-h) - c) / c
        return res

    @classmethod
    def profile_condition(
        cls,
        df: pd.DataFrame,
        condition_mask: pd.Series,
        condition_name: str,
        horizons: list[int] | None = None,
    ) -> dict[int, StatisticalProfile]:
        """Profile forward outcomes whenever `condition_mask` is True."""
        if horizons is None:
            horizons = [1, 3, 5, 10]

        df_fwd = cls.calculate_forward_returns(df, horizons)
        filtered = df_fwd[condition_mask]

        profiles: dict[int, StatisticalProfile] = {}

        for h in horizons:
            col = f"fwd_return_{h}"
            series = filtered[col].dropna()
            n = len(series)

            if n < 5:
                profiles[h] = StatisticalProfile(
                    condition_name=condition_name,
                    horizon_bars=h,
                    sample_size=n,
                )
                continue

            vals = series.values
            wins = vals[vals > 0]
            losses = vals[vals < 0]

            win_count = len(wins)
            loss_count = len(losses)
            win_rate = win_count / n
            loss_rate = loss_count / n

            mean_ret = float(np.mean(vals))
            median_ret = float(np.median(vals))
            std_ret = float(np.std(vals, ddof=1)) if n > 1 else 0.0

            mean_w = float(np.mean(wins)) if win_count > 0 else 0.0
            mean_l = float(abs(np.mean(losses))) if loss_count > 0 else 0.0
            payoff = (mean_w / mean_l) if mean_l > 0 else 0.0

            expectancy = (win_rate * mean_w) - (loss_rate * mean_l)

            sum_wins = float(np.sum(wins)) if win_count > 0 else 0.0
            sum_losses = float(abs(np.sum(losses))) if loss_count > 0 else 0.0
            profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (99.0 if sum_wins > 0 else 0.0)

            # Higher moments
            skew = float(pd.Series(vals).skew())
            kurt = float(pd.Series(vals).kurt())

            # T-statistic against mean == 0
            se = std_ret / np.sqrt(n) if std_ret > 0 else 1e-8
            t_stat = mean_ret / se

            profiles[h] = StatisticalProfile(
                condition_name=condition_name,
                horizon_bars=h,
                sample_size=n,
                win_rate=round(win_rate, 4),
                loss_rate=round(loss_rate, 4),
                mean_return=round(mean_ret, 5),
                median_return=round(median_ret, 5),
                mean_win=round(mean_w, 5),
                mean_loss=round(mean_l, 5),
                payoff_ratio=round(payoff, 4),
                expectancy=round(expectancy, 5),
                profit_factor=round(profit_factor, 4),
                std_dev=round(std_ret, 5),
                skewness=round(skew, 4),
                kurtosis=round(kurt, 4),
                t_stat=round(t_stat, 3),
            )

        return profiles
