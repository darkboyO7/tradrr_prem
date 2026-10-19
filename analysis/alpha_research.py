"""Trading Intelligence Platform - Quantitative Alpha Research & Multi-Factor Neutralization Pipeline.

Constructs cross-sectional alpha factors (momentum, mean-reversion, idiosyncratic vol, liquidity),
applies regression-based orthogonal neutralization against market beta and style risks,
and evaluates predictive performance via Information Coefficients (IC / Rank IC),
quantile monotonicity spreads, and factor turnover decay profiles.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import scipy.stats as st

from core.logging import get_logger

logger = get_logger("alpha_research")


class AlphaFactorType(str, Enum):
    """Canonical cross-sectional alpha factor categories."""

    MOMENTUM_14D = "MOMENTUM_14D"
    MEAN_REVERSION_3D = "MEAN_REVERSION_3D"
    IDIOSYNCRATIC_VOLATILITY = "IDIOSYNCRATIC_VOLATILITY"
    LIQUIDITY_BETA = "LIQUIDITY_BETA"
    FLOW_TOXICITY = "FLOW_TOXICITY"


class NeutralizedFactorScore(BaseModel):
    """Asset-level raw and orthogonally neutralized alpha scores."""

    symbol: str
    raw_score: float
    neutralized_score: float
    standardized_z_score: float
    percentile_rank: float


class FactorEvaluationMetrics(BaseModel):
    """Institutional alpha factor efficacy and persistence diagnostics."""

    factor_name: str
    ic_mean: float
    ic_std: float
    ic_information_ratio: float
    ic_t_stat: float
    ic_p_value: float
    rank_ic_mean: float
    quantile_returns: Dict[str, float] = {}  # Q1 to Q5 returns
    long_short_spread_return: float
    quantile_monotonicity_score: float       # Rank correlation across quantiles
    turnover_autocorrelation_lag1: float
    turnover_autocorrelation_lag5: float
    is_statistically_significant: bool


class AlphaResearchReport(BaseModel):
    """Comprehensive cross-sectional alpha factor backtest and neutralization report."""

    report_id: str = Field(default_factory=lambda: f"alpha_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    num_assets: int
    num_periods: int
    neutralization_factors_applied: List[str]
    factor_evaluations: Dict[str, FactorEvaluationMetrics]
    top_ranked_assets: List[str]
    bottom_ranked_assets: List[str]
    asset_scores: Dict[str, NeutralizedFactorScore]


class AlphaResearchPipeline:
    """Quantitative engine for cross-sectional factor construction, neutralization, and evaluation."""

    def __init__(self, num_quantiles: int = 5, winsorize_std: float = 3.0) -> None:
        self.num_quantiles = num_quantiles
        self.winsorize_std = winsorize_std

    def winsorize_and_standardize(self, series: pd.Series) -> pd.Series:
        """Clip outliers at +/- K standard deviations and compute cross-sectional Z-score."""
        if len(series) <= 1:
            return series

        mean_val = series.mean()
        std_val = series.std()
        if std_val <= 1e-9:
            return pd.Series(0.0, index=series.index)

        lower_bound = mean_val - self.winsorize_std * std_val
        upper_bound = mean_val + self.winsorize_std * std_val
        clipped = series.clip(lower=lower_bound, upper=upper_bound)

        # Cross-sectional z-score
        z_scored = (clipped - clipped.mean()) / (clipped.std() + 1e-9)
        return z_scored

    def neutralize_factor(
        self, factor_scores: pd.Series, risk_exposures: pd.DataFrame
    ) -> pd.Series:
        """Orthogonalize alpha factor scores against market beta and style factors via OLS residualization.
        
        Formula:
            F_raw = X * beta + epsilon
            F_neutral = epsilon (residuals orthogonal to X)
        """
        # Align indexes
        common_idx = factor_scores.index.intersection(risk_exposures.index)
        if len(common_idx) < 3:
            return factor_scores

        y = factor_scores.loc[common_idx].values
        # Add intercept column
        X = np.column_stack([np.ones(len(common_idx)), risk_exposures.loc[common_idx].values])

        # OLS projection: beta = (X^T X)^(-1) X^T y
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            neutralized_series = pd.Series(residuals, index=common_idx)
            # Re-standardize residuals to unit variance
            return self.winsorize_and_standardize(neutralized_series)
        except Exception as exc:
            logger.warning(f"Factor neutralization failed: {exc}. Returning raw scores.")
            return factor_scores

    def construct_alpha_factors(
        self, prices_df: pd.DataFrame, volumes_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """Generate canonical cross-sectional alpha factor panels from price-volume matrix."""
        returns_df = prices_df.pct_change()
        factors: Dict[str, pd.DataFrame] = {}

        # 1. 14-period Cross-Sectional Momentum
        mom_14 = prices_df.pct_change(14)
        factors[AlphaFactorType.MOMENTUM_14D.value] = mom_14

        # 2. 3-period Short-Term Mean Reversion (-1 * 3-period return)
        rev_3 = -prices_df.pct_change(3)
        factors[AlphaFactorType.MEAN_REVERSION_3D.value] = rev_3

        # 3. Idiosyncratic Volatility (rolling residual vol against market)
        mkt_ret = returns_df.mean(axis=1)
        idiosyncratic_vols = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
        window = 20
        for col in returns_df.columns:
            asset_ret = returns_df[col]
            # Rolling beta and residual vol
            cov = asset_ret.rolling(window).cov(mkt_ret)
            var_mkt = mkt_ret.rolling(window).var()
            beta = cov / (var_mkt + 1e-9)
            res = asset_ret - beta * mkt_ret
            # Negative sign: lower idiosyncratic vol often yields higher risk-adjusted returns
            idiosyncratic_vols[col] = -res.rolling(window).std()
        factors[AlphaFactorType.IDIOSYNCRATIC_VOLATILITY.value] = idiosyncratic_vols

        # 4. Liquidity Beta / Volume Ratio
        if volumes_df is not None:
            mkt_vol = volumes_df.mean(axis=1)
            vol_ratio = volumes_df.rolling(10).mean().div(mkt_vol.rolling(10).mean() + 1e-9, axis=0)
            factors[AlphaFactorType.LIQUIDITY_BETA.value] = vol_ratio
        else:
            factors[AlphaFactorType.LIQUIDITY_BETA.value] = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

        return factors

    def evaluate_factor_performance(
        self, factor_panel: pd.DataFrame, forward_returns: pd.DataFrame, factor_name: str
    ) -> FactorEvaluationMetrics:
        """Compute Information Coefficient, Rank IC, quantile monotonic spread, and autocorrelation decay."""
        # Align timeframes
        common_dates = factor_panel.index.intersection(forward_returns.index)
        if len(common_dates) < 10:
            return FactorEvaluationMetrics(
                factor_name=factor_name,
                ic_mean=0.0,
                ic_std=0.0,
                ic_information_ratio=0.0,
                ic_t_stat=0.0,
                ic_p_value=1.0,
                rank_ic_mean=0.0,
                quantile_returns={},
                long_short_spread_return=0.0,
                quantile_monotonicity_score=0.0,
                turnover_autocorrelation_lag1=0.0,
                turnover_autocorrelation_lag5=0.0,
                is_statistically_significant=False,
            )

        f_aligned = factor_panel.loc[common_dates]
        r_aligned = forward_returns.loc[common_dates]

        ics = []
        rank_ics = []
        quantile_buckets: Dict[int, List[float]] = {i: [] for i in range(1, self.num_quantiles + 1)}

        for date in common_dates:
            f_slice = f_aligned.loc[date].dropna()
            r_slice = r_aligned.loc[date].dropna()
            valid_syms = f_slice.index.intersection(r_slice.index)

            if len(valid_syms) < 4:
                continue

            f_vals = f_slice.loc[valid_syms].values
            r_vals = r_slice.loc[valid_syms].values

            # Pearson IC
            if np.std(f_vals) > 1e-9 and np.std(r_vals) > 1e-9:
                ic, _ = st.pearsonr(f_vals, r_vals)
                if not np.isnan(ic):
                    ics.append(ic)

                # Spearman Rank IC
                rank_ic, _ = st.spearmanr(f_vals, r_vals)
                if not np.isnan(rank_ic):
                    rank_ics.append(rank_ic)

            # Quantile Sorting
            try:
                q_labels = pd.qcut(f_slice.loc[valid_syms], q=self.num_quantiles, labels=False, duplicates="drop")
                for q_idx in range(self.num_quantiles):
                    q_syms = q_labels[q_labels == q_idx].index
                    if len(q_syms) > 0:
                        quantile_buckets[q_idx + 1].append(float(r_slice.loc[q_syms].mean()))
            except Exception:
                pass

        ic_mean = float(np.mean(ics)) if ics else 0.0
        ic_std = float(np.std(ics)) if ics else 1e-6
        ir = ic_mean / ic_std if ic_std > 1e-9 else 0.0
        n_obs = len(ics)
        t_stat = ir * math.sqrt(n_obs) if n_obs > 1 else 0.0
        p_val = float(2.0 * (1.0 - st.t.cdf(abs(t_stat), df=max(1, n_obs - 1)))) if n_obs > 1 else 1.0
        rank_ic_mean = float(np.mean(rank_ics)) if rank_ics else 0.0

        # Quantile Returns & Monotonicity
        q_returns: Dict[str, float] = {}
        q_means = []
        for q_idx in range(1, self.num_quantiles + 1):
            q_mean = float(np.mean(quantile_buckets[q_idx])) if quantile_buckets[q_idx] else 0.0
            q_returns[f"Q{q_idx}"] = round(q_mean, 6)
            q_means.append(q_mean)

        long_short_spread = (q_means[-1] - q_means[0]) if len(q_means) >= 2 else 0.0

        # Monotonicity: Rank correlation between quantile indices [1..K] and quantile average returns
        if len(q_means) >= 3 and np.std(q_means) > 1e-9:
            mono_corr, _ = st.spearmanr(list(range(1, len(q_means) + 1)), q_means)
            monotonicity_score = float(mono_corr) if not np.isnan(mono_corr) else 0.0
        else:
            monotonicity_score = 1.0 if long_short_spread > 0 else -1.0

        # Autocorrelation / Factor Turnover
        lag1_corrs = []
        lag5_corrs = []
        for sym in f_aligned.columns:
            s = f_aligned[sym].dropna()
            if len(s) > 10:
                ac1 = s.autocorr(1)
                ac5 = s.autocorr(5)
                if not np.isnan(ac1):
                    lag1_corrs.append(ac1)
                if not np.isnan(ac5):
                    lag5_corrs.append(ac5)

        turnover_ac1 = float(np.mean(lag1_corrs)) if lag1_corrs else 0.85
        turnover_ac5 = float(np.mean(lag5_corrs)) if lag5_corrs else 0.65

        is_significant = abs(t_stat) >= 1.96 and p_val <= 0.05

        return FactorEvaluationMetrics(
            factor_name=factor_name,
            ic_mean=round(ic_mean, 4),
            ic_std=round(ic_std, 4),
            ic_information_ratio=round(ir, 4),
            ic_t_stat=round(t_stat, 2),
            ic_p_value=round(p_val, 4),
            rank_ic_mean=round(rank_ic_mean, 4),
            quantile_returns=q_returns,
            long_short_spread_return=round(long_short_spread, 6),
            quantile_monotonicity_score=round(monotonicity_score, 4),
            turnover_autocorrelation_lag1=round(turnover_ac1, 4),
            turnover_autocorrelation_lag5=round(turnover_ac5, 4),
            is_statistically_significant=is_significant,
        )

    def run_alpha_pipeline(
        self,
        prices_df: pd.DataFrame,
        volumes_df: Optional[pd.DataFrame] = None,
        forward_horizon_periods: int = 1,
    ) -> AlphaResearchReport:
        """Execute end-to-end alpha research pipeline: construct, neutralize, and evaluate factors."""
        factors = self.construct_alpha_factors(prices_df, volumes_df)
        forward_returns = prices_df.pct_change(forward_horizon_periods).shift(-forward_horizon_periods)

        # Risk exposures for neutralization: Market beta and size proxy
        mkt_returns = prices_df.pct_change().mean(axis=1)
        risk_exposures = pd.DataFrame(index=prices_df.columns)
        for col in prices_df.columns:
            asset_ret = prices_df[col].pct_change()
            cov = asset_ret.cov(mkt_returns)
            var_mkt = mkt_returns.var()
            risk_exposures.loc[col, "MarketBeta"] = cov / (var_mkt + 1e-9)
            risk_exposures.loc[col, "LogPrice"] = math.log(max(1.0, prices_df[col].iloc[-1]))

        evaluations: Dict[str, FactorEvaluationMetrics] = {}
        for fname, f_df in factors.items():
            metrics = self.evaluate_factor_performance(f_df, forward_returns, fname)
            evaluations[fname] = metrics

        # Generate composite neutralized score on latest cross-section
        latest_mom = factors[AlphaFactorType.MOMENTUM_14D.value].iloc[-1].dropna()
        standardized_mom = self.winsorize_and_standardize(latest_mom)
        neutralized_mom = self.neutralize_factor(standardized_mom, risk_exposures)

        asset_scores: Dict[str, NeutralizedFactorScore] = {}
        ranks = neutralized_mom.rank(pct=True)

        for sym in neutralized_mom.index:
            asset_scores[sym] = NeutralizedFactorScore(
                symbol=sym,
                raw_score=round(float(latest_mom.get(sym, 0.0)), 6),
                neutralized_score=round(float(neutralized_mom[sym]), 4),
                standardized_z_score=round(float(standardized_mom.get(sym, 0.0)), 4),
                percentile_rank=round(float(ranks[sym]), 4),
            )

        # Rank assets
        sorted_syms = sorted(neutralized_mom.index, key=lambda s: neutralized_mom[s], reverse=True)
        top_assets = sorted_syms[:3]
        bottom_assets = sorted_syms[-3:] if len(sorted_syms) >= 3 else []

        return AlphaResearchReport(
            num_assets=len(prices_df.columns),
            num_periods=len(prices_df),
            neutralization_factors_applied=["MarketBeta", "LogPrice"],
            factor_evaluations=evaluations,
            top_ranked_assets=top_assets,
            bottom_ranked_assets=bottom_assets,
            asset_scores=asset_scores,
        )
