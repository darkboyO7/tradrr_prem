"""Trading Intelligence Platform - Dynamic Volatility Targeting Engine.

Dynamically scales portfolio gross leverage and asset weights to maintain a constant
target annualized volatility (e.g., 15%), systematically reducing exposure during
market stress regimes and expanding allocation during low-volatility regimes.
"""

from datetime import datetime, timezone
import math
from typing import Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("vol_targeting")


class VolatilityTargetConfig(BaseModel):
    """Configuration parameters for dynamic volatility targeting."""

    target_volatility_annualized: float = Field(
        default=0.15,
        ge=0.02,
        le=0.60,
        description="Institutional annualized target portfolio volatility (e.g., 15%)",
    )
    min_leverage: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum gross exposure floor during extreme volatility spikes",
    )
    max_leverage: float = Field(
        default=1.50,
        ge=0.5,
        le=3.0,
        description="Maximum permissible gross leverage ceiling during subdued volatility",
    )
    volatility_floor: float = Field(
        default=0.04,
        ge=0.01,
        description="Absolute minimum realized volatility bound to prevent division blowup",
    )
    ewma_decay_lambda: float = Field(
        default=0.94,
        ge=0.80,
        le=0.99,
        description="RiskMetrics exponential decay factor for realized volatility estimation",
    )


class VolatilityTargetResult(BaseModel):
    """Output metrics from dynamic volatility targeting scaling."""

    target_volatility_annualized: float
    realized_volatility_annualized: float
    unconstrained_multiplier: float
    applied_multiplier: float
    gross_exposure_pct: float
    cash_buffer_pct: float
    base_weights: Dict[str, float]
    target_weights: Dict[str, float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VolatilityTargetingEngine:
    """Institutional portfolio volatility scaling engine."""

    def __init__(self, config: Optional[VolatilityTargetConfig] = None) -> None:
        self.config = config or VolatilityTargetConfig()

    def estimate_portfolio_volatility(
        self,
        prices_df: pd.DataFrame,
        weights: Dict[str, float],
    ) -> float:
        """Estimate current annualized realized portfolio volatility using EWMA covariance."""
        symbols = [s for s in weights.keys() if s in prices_df.columns]
        if not symbols:
            raise ValueError("No matching asset symbols found in price history")

        # Normalize weights
        total_w = sum(weights[s] for s in symbols)
        if total_w <= 0:
            raise ValueError("Sum of portfolio weights must be positive")
        w = np.array([weights[s] / total_w for s in symbols])

        # Compute log returns
        returns = np.log(prices_df[symbols] / prices_df[symbols].shift(1)).dropna()
        if len(returns) < 5:
            raise ValueError(f"Insufficient price history for volatility calculation (need >= 5, got {len(returns)})")

        t = len(returns)
        lam = self.config.ewma_decay_lambda
        decay_weights = np.array([(1.0 - lam) * (lam ** (t - 1 - i)) for i in range(t)])
        decay_weights /= decay_weights.sum()

        mean_adj = returns - returns.mean()
        weighted_cov = np.zeros((len(symbols), len(symbols)))
        for i in range(t):
            row = mean_adj.iloc[i].values.reshape(-1, 1)
            weighted_cov += decay_weights[i] * (row @ row.T)

        daily_variance = float(w.T @ weighted_cov @ w)
        daily_vol = math.sqrt(max(daily_variance, 1e-12))
        annualized_vol = daily_vol * math.sqrt(365.0)  # 24/7 crypto annualization

        return max(annualized_vol, self.config.volatility_floor)

    def calculate_target_weights(
        self,
        prices_df: pd.DataFrame,
        base_weights: Dict[str, float],
    ) -> VolatilityTargetResult:
        """Compute optimal volatility-targeted portfolio weights and leverage scaling.
        
        Formula:
            Multiplier = Target_Vol / Realized_Vol
            Applied_Multiplier = clip(Multiplier, min_leverage, max_leverage)
            Target_Weight_i = Base_Weight_i * Applied_Multiplier
        """
        realized_vol = self.estimate_portfolio_volatility(prices_df, base_weights)

        unconstrained_mult = self.config.target_volatility_annualized / realized_vol
        applied_mult = max(
            self.config.min_leverage,
            min(self.config.max_leverage, unconstrained_mult),
        )

        target_weights = {
            sym: round(weight * applied_mult, 6)
            for sym, weight in base_weights.items()
        }

        gross_exposure = sum(abs(w) for w in target_weights.values())
        cash_buffer = max(0.0, 1.0 - gross_exposure)

        logger.info(
            "Volatility targeting evaluated",
            target_vol=self.config.target_volatility_annualized,
            realized_vol=round(realized_vol, 4),
            multiplier=round(applied_mult, 4),
            gross_exposure=round(gross_exposure, 4),
        )

        return VolatilityTargetResult(
            target_volatility_annualized=self.config.target_volatility_annualized,
            realized_volatility_annualized=round(realized_vol, 4),
            unconstrained_multiplier=round(unconstrained_mult, 4),
            applied_multiplier=round(applied_mult, 4),
            gross_exposure_pct=round(gross_exposure * 100.0, 2),
            cash_buffer_pct=round(cash_buffer * 100.0, 2),
            base_weights=base_weights,
            target_weights=target_weights,
        )
