"""Trading Intelligence Platform - Multi-Asset Covariance & Value-at-Risk (VaR) Engine.

Calculates multi-asset covariance matrices (sample and exponentially weighted),
Parametric VaR, Historical Simulation VaR, Monte Carlo VaR, Expected Shortfall (CVaR),
and Component/Marginal VaR risk decomposition across institutional portfolio assets.
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("var_model")


class VaRMetrics(BaseModel):
    """Calculated Value-at-Risk and Expected Shortfall metrics across multiple methodologies."""

    portfolio_value: float
    confidence_level: float
    horizon_days: int
    portfolio_volatility_daily: float
    portfolio_volatility_annualized: float

    # VaR dollar values (positive numbers representing potential loss)
    parametric_var: float
    historical_var: float
    monte_carlo_var: float

    # Expected Shortfall (Conditional VaR)
    parametric_cvar: float
    historical_cvar: float
    monte_carlo_cvar: float

    # Asset-level risk attribution
    component_var: Dict[str, float] = Field(
        default_factory=dict,
        description="Dollar risk contribution per asset (sums to total parametric VaR)",
    )
    percentage_risk_contribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of total portfolio risk contributed by each asset",
    )


class PortfolioRiskModel:
    """Quantitative risk engine computing covariance, VaR, and risk decomposition."""

    # Standard normal quantile approximations
    Z_SCORES = {
        0.90: 1.28155,
        0.95: 1.64485,
        0.975: 1.95996,
        0.99: 2.32635,
        0.999: 3.09023,
    }

    def __init__(self, decay_factor: float = 0.94) -> None:
        """Initialize risk model.
        
        Args:
            decay_factor: Lambda for RiskMetrics exponentially weighted covariance (e.g. 0.94).
        """
        self.decay_factor = decay_factor

    def compute_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Compute logarithmic returns from asset price history."""
        returns = np.log(prices_df / prices_df.shift(1)).dropna()
        return returns

    def compute_covariance_matrix(
        self,
        returns: pd.DataFrame,
        exponential_weighting: bool = True,
    ) -> pd.DataFrame:
        """Compute sample or exponentially-weighted covariance matrix."""
        if not exponential_weighting or len(returns) < 5:
            return returns.cov()

        # RiskMetrics EWMA covariance
        t = len(returns)
        weights = np.array([(1.0 - self.decay_factor) * (self.decay_factor ** (t - 1 - i)) for i in range(t)])
        weights /= weights.sum()

        mean_adjusted = returns - returns.mean()
        weighted_cov = np.zeros((returns.shape[1], returns.shape[1]))

        for i in range(t):
            row = mean_adjusted.iloc[i].values.reshape(-1, 1)
            weighted_cov += weights[i] * (row @ row.T)

        return pd.DataFrame(weighted_cov, index=returns.columns, columns=returns.columns)

    def calculate_var(
        self,
        prices_df: pd.DataFrame,
        weights: Dict[str, float],
        portfolio_value: float = 100_000.0,
        confidence_level: float = 0.95,
        horizon_days: int = 1,
        monte_carlo_sims: int = 10_000,
        random_seed: int = 42,
    ) -> VaRMetrics:
        """Compute multi-methodology VaR, CVaR, and component risk decomposition.
        
        Args:
            prices_df: DataFrame with asset prices as columns.
            weights: Dict mapping symbol -> portfolio weight (e.g. {"BTC-USD": 0.6, "ETH-USD": 0.4}).
            portfolio_value: Current total equity USD.
            confidence_level: VaR confidence interval (0.95 or 0.99).
            horizon_days: Forecast risk horizon in trading days.
            monte_carlo_sims: Number of Monte Carlo return simulation paths.
            random_seed: Deterministic seed for reproducible simulation.
        """
        # Clean and align assets
        symbols = [s for s in weights.keys() if s in prices_df.columns]
        if not symbols:
            raise ValueError("No matching symbols between portfolio weights and price DataFrame")

        total_weight = sum(weights[s] for s in symbols)
        if total_weight <= 0:
            raise ValueError("Total portfolio weight must be positive")

        # Normalize weights
        w = np.array([weights[s] / total_weight for s in symbols])
        sub_prices = prices_df[symbols]
        returns = self.compute_returns(sub_prices)

        if len(returns) < 10:
            raise ValueError(f"Insufficient price history: need at least 10 return bars, got {len(returns)}")

        cov_matrix = self.compute_covariance_matrix(returns, exponential_weighting=True)
        cov_np = cov_matrix.values

        # 1. Portfolio Volatility
        port_variance_daily = float(w.T @ cov_np @ w)
        port_vol_daily = math.sqrt(max(port_variance_daily, 1e-12))
        port_vol_horizon = port_vol_daily * math.sqrt(horizon_days)
        port_vol_annualized = port_vol_daily * math.sqrt(365.0)  # 24/7 crypto annualization

        # Z-score for confidence level
        z = self.Z_SCORES.get(confidence_level, 1.64485)

        # 2. Parametric VaR and CVaR
        # VaR = Z * sigma * Value
        parametric_var = round(z * port_vol_horizon * portfolio_value, 2)
        # CVaR (Expected Shortfall for normal dist): phi(z)/(1 - alpha) * sigma * Value
        norm_pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * (z ** 2))
        tail_prob = 1.0 - confidence_level
        cvar_multiplier = norm_pdf / tail_prob if tail_prob > 0 else z
        parametric_cvar = round(cvar_multiplier * port_vol_horizon * portfolio_value, 2)

        # 3. Component & Marginal VaR (Euler Decomposition)
        # Marginal VaR = Z * (Cov * w) / port_vol * horizon_factor
        cov_w = cov_np @ w
        marginal_var_factors = z * (cov_w / port_vol_daily) * math.sqrt(horizon_days)
        component_var_dollars = {}
        pct_risk_contrib = {}

        for i, sym in enumerate(symbols):
            comp_var = round(float(w[i] * marginal_var_factors[i] * portfolio_value), 2)
            component_var_dollars[sym] = comp_var
            pct_contrib = round((comp_var / parametric_var) * 100.0, 2) if parametric_var > 0 else 0.0
            pct_risk_contrib[sym] = pct_contrib

        # 4. Historical Simulation VaR
        # Calculate empirical portfolio historical returns
        port_hist_returns = (returns[symbols].values @ w) * math.sqrt(horizon_days)
        hist_cutoff_quantile = (1.0 - confidence_level) * 100.0
        hist_var_return = -np.percentile(port_hist_returns, hist_cutoff_quantile)
        historical_var = round(float(max(hist_var_return * portfolio_value, 0.0)), 2)

        # Historical CVaR: average of losses exceeding the VaR cutoff
        tail_losses = port_hist_returns[port_hist_returns <= -hist_var_return]
        if len(tail_losses) > 0:
            historical_cvar = round(float(max(-tail_losses.mean() * portfolio_value, historical_var)), 2)
        else:
            historical_cvar = historical_var

        # 5. Monte Carlo Simulation VaR
        rng = np.random.default_rng(random_seed)
        # Cholesky decomposition: Sigma = L * L.T
        try:
            # Add micro-epsilon jitter to ensure positive definiteness
            jitter = np.eye(len(symbols)) * 1e-9
            L = np.linalg.cholesky(cov_np + jitter)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition if covariance is singular
            eigvals, eigvecs = np.linalg.eigh(cov_np)
            eigvals = np.maximum(eigvals, 1e-8)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        std_normals = rng.standard_normal((monte_carlo_sims, len(symbols)))
        simulated_returns = (std_normals @ L.T) * math.sqrt(horizon_days)
        sim_port_returns = simulated_returns @ w

        mc_cutoff = (1.0 - confidence_level) * 100.0
        mc_var_return = -np.percentile(sim_port_returns, mc_cutoff)
        monte_carlo_var = round(float(max(mc_var_return * portfolio_value, 0.0)), 2)

        mc_tail = sim_port_returns[sim_port_returns <= -mc_var_return]
        if len(mc_tail) > 0:
            monte_carlo_cvar = round(float(max(-mc_tail.mean() * portfolio_value, monte_carlo_var)), 2)
        else:
            monte_carlo_cvar = monte_carlo_var

        return VaRMetrics(
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
            horizon_days=horizon_days,
            portfolio_volatility_daily=round(port_vol_daily, 6),
            portfolio_volatility_annualized=round(port_vol_annualized, 4),
            parametric_var=parametric_var,
            historical_var=historical_var,
            monte_carlo_var=monte_carlo_var,
            parametric_cvar=parametric_cvar,
            historical_cvar=historical_cvar,
            monte_carlo_cvar=monte_carlo_cvar,
            component_var=component_var_dollars,
            percentage_risk_contribution=pct_risk_contrib,
        )
