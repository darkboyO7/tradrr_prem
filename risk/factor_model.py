"""Trading Intelligence Platform - Quantitative Factor Attribution Model.

Decomposes asset and portfolio returns into systematic institutional risk factors:
Market Beta, Cross-Sectional Momentum, Volatility Exposure, and Liquidity/Size Factors,
isolating pure idiosyncratic alpha and computing multi-factor risk attribution.
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("factor_model")


class AssetFactorExposure(BaseModel):
    """Systematic factor loadings and attribution for an individual asset."""

    symbol: str
    alpha_annualized: float = Field(description="Annualized idiosyncratic excess return")
    r_squared: float = Field(description="Percentage of return variance explained by systematic factors")
    specific_risk_annualized: float = Field(description="Idiosyncratic volatility unexplained by factors")
    factor_betas: Dict[str, float] = Field(description="Factor loadings: Market, Momentum, Volatility, Liquidity")
    factor_variance_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Share of total variance driven by each systematic factor",
    )


class PortfolioFactorAttribution(BaseModel):
    """Aggregate portfolio systematic factor decomposition."""

    portfolio_r_squared: float
    portfolio_beta_market: float
    portfolio_beta_momentum: float
    portfolio_beta_volatility: float
    portfolio_beta_liquidity: float
    systematic_risk_pct: float
    idiosyncratic_risk_pct: float
    asset_exposures: Dict[str, AssetFactorExposure] = Field(default_factory=dict)


class SystematicFactorModel:
    """Multi-factor regression model based on Arbitrage Pricing Theory (APT)."""

    FACTOR_NAMES = ["Market", "Momentum", "Volatility", "Liquidity"]

    def __init__(self, benchmark_symbol: str = "BTC-USD") -> None:
        self.benchmark_symbol = benchmark_symbol.upper()

    def construct_synthetic_factors(self, prices_df: pd.DataFrame, volumes_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Construct canonical systemic factor return series from historical asset prices and volumes.
        
        Factors:
            - Market: Returns of benchmark_symbol (or equal-weighted basket if benchmark absent)
            - Momentum: Long top 30% rolling 14-period performers, short bottom 30%
            - Volatility: Long high-volatility assets, short low-volatility assets
            - Liquidity: Long high-volume assets, short low-volume assets
        """
        returns = np.log(prices_df / prices_df.shift(1)).dropna()
        if len(returns) < 15:
            raise ValueError(f"Insufficient history for factor construction: need >= 15 bars, got {len(returns)}")

        factors_df = pd.DataFrame(index=returns.index)

        # 1. Market Factor
        if self.benchmark_symbol in returns.columns:
            factors_df["Market"] = returns[self.benchmark_symbol]
        else:
            factors_df["Market"] = returns.mean(axis=1)

        # 2. Momentum Factor (14-period momentum spread)
        mom_14 = (prices_df / prices_df.shift(14) - 1.0).reindex(returns.index).fillna(0.0)
        # Compute cross-sectional rankings per row
        mom_ranked = mom_14.rank(axis=1, pct=True)
        high_mom = (mom_ranked >= 0.7).astype(float)
        low_mom = (mom_ranked <= 0.3).astype(float)
        high_mom_weights = high_mom.div(high_mom.sum(axis=1).replace(0, 1), axis=0)
        low_mom_weights = low_mom.div(low_mom.sum(axis=1).replace(0, 1), axis=0)
        factors_df["Momentum"] = (returns * high_mom_weights).sum(axis=1) - (returns * low_mom_weights).sum(axis=1)

        # 3. Volatility Factor (rolling 14-period standard deviation spread)
        vol_14 = returns.rolling(window=14).std().bfill().fillna(0.0)
        vol_ranked = vol_14.rank(axis=1, pct=True)
        high_vol = (vol_ranked >= 0.7).astype(float)
        low_vol = (vol_ranked <= 0.3).astype(float)
        high_vol_weights = high_vol.div(high_vol.sum(axis=1).replace(0, 1), axis=0)
        low_vol_weights = low_vol.div(low_vol.sum(axis=1).replace(0, 1), axis=0)
        factors_df["Volatility"] = (returns * high_vol_weights).sum(axis=1) - (returns * low_vol_weights).sum(axis=1)

        # 4. Liquidity Factor
        if volumes_df is not None and not volumes_df.empty:
            aligned_vols = volumes_df.reindex(returns.index).fillna(1.0)
            vol_rank = aligned_vols.rank(axis=1, pct=True)
            high_liq = (vol_rank >= 0.7).astype(float)
            low_liq = (vol_rank <= 0.3).astype(float)
            factors_df["Liquidity"] = (returns * high_liq.div(high_liq.sum(axis=1).replace(0, 1), axis=0)).sum(axis=1) - \
                                      (returns * low_liq.div(low_liq.sum(axis=1).replace(0, 1), axis=0)).sum(axis=1)
        else:
            # Synthetic size proxy: mean return minus market return
            factors_df["Liquidity"] = returns.mean(axis=1) - factors_df["Market"]

        return factors_df

    def analyze_asset(
        self,
        asset_returns: pd.Series,
        factors_df: pd.DataFrame,
        symbol: str,
    ) -> AssetFactorExposure:
        """Run multi-variable OLS regression to estimate factor loadings."""
        # Align series
        aligned = pd.concat([asset_returns.rename("asset"), factors_df], axis=1).dropna()
        y = aligned["asset"].values
        X_factors = aligned[self.FACTOR_NAMES].values
        # Add intercept for alpha
        X = np.column_stack([np.ones(len(y)), X_factors])

        # OLS closed-form solution: beta = (X^T X)^-1 X^T y
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            betas = np.zeros(X.shape[1])

        alpha_daily = betas[0]
        alpha_annual = round(float(alpha_daily * 365.0), 4)

        factor_betas = {}
        for idx, name in enumerate(self.FACTOR_NAMES):
            factor_betas[name] = round(float(betas[idx + 1]), 4)

        # Fitted values & residuals
        y_pred = X @ betas
        residuals = y - y_pred
        total_var = float(np.var(y))
        residual_var = float(np.var(residuals))

        r_squared = round(float(1.0 - (residual_var / total_var)), 4) if total_var > 1e-12 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        specific_risk_daily = math.sqrt(max(residual_var, 0.0))
        specific_risk_annual = round(specific_risk_daily * math.sqrt(365.0), 4)

        # Factor variance contribution
        # Var(y_pred) = sum_k beta_k * Cov(F_k, y)
        cov_matrix = np.cov(X_factors.T)
        factor_b = betas[1:]
        factor_explained_var = factor_b.T @ cov_matrix @ factor_b if cov_matrix.ndim == 2 else 0.0

        var_contribs = {}
        for idx, name in enumerate(self.FACTOR_NAMES):
            contrib = (factor_b[idx] ** 2) * (cov_matrix[idx, idx] if cov_matrix.ndim == 2 else cov_matrix)
            pct = round(float(contrib / total_var * 100.0), 2) if total_var > 0 else 0.0
            var_contribs[name] = pct

        return AssetFactorExposure(
            symbol=symbol,
            alpha_annualized=alpha_annual,
            r_squared=r_squared,
            specific_risk_annualized=specific_risk_annual,
            factor_betas=factor_betas,
            factor_variance_contributions=var_contribs,
        )

    def attribute_portfolio(
        self,
        prices_df: pd.DataFrame,
        weights: Dict[str, float],
        volumes_df: Optional[pd.DataFrame] = None,
    ) -> PortfolioFactorAttribution:
        """Calculate systematic factor attribution across the full portfolio."""
        factors_df = self.construct_synthetic_factors(prices_df, volumes_df)
        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

        symbols = [s for s in weights.keys() if s in returns_df.columns]
        total_weight = sum(weights[s] for s in symbols)
        norm_weights = {s: weights[s] / total_weight for s in symbols}

        asset_exposures: Dict[str, AssetFactorExposure] = {}
        port_beta_mkt = 0.0
        port_beta_mom = 0.0
        port_beta_vol = 0.0
        port_beta_liq = 0.0
        port_weighted_r2 = 0.0

        for sym in symbols:
            exp = self.analyze_asset(returns_df[sym], factors_df, symbol=sym)
            asset_exposures[sym] = exp
            w = norm_weights[sym]
            port_beta_mkt += w * exp.factor_betas.get("Market", 0.0)
            port_beta_mom += w * exp.factor_betas.get("Momentum", 0.0)
            port_beta_vol += w * exp.factor_betas.get("Volatility", 0.0)
            port_beta_liq += w * exp.factor_betas.get("Liquidity", 0.0)
            port_weighted_r2 += w * exp.r_squared

        systematic_pct = round(port_weighted_r2 * 100.0, 2)
        idiosyncratic_pct = round(100.0 - systematic_pct, 2)

        return PortfolioFactorAttribution(
            portfolio_r_squared=round(port_weighted_r2, 4),
            portfolio_beta_market=round(port_beta_mkt, 4),
            portfolio_beta_momentum=round(port_beta_mom, 4),
            portfolio_beta_volatility=round(port_beta_vol, 4),
            portfolio_beta_liquidity=round(port_beta_liq, 4),
            systematic_risk_pct=systematic_pct,
            idiosyncratic_risk_pct=idiosyncratic_pct,
            asset_exposures=asset_exposures,
        )
