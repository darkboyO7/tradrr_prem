"""Trading Intelligence Platform - Risk Limits & Portfolio Constraints.

Defines firm-level risk thresholds, single-asset maximum exposure,
maximum leverage, and drawdown circuit breaker triggers.
"""

from pydantic import BaseModel, Field


class RiskLimits(BaseModel):
    """Firm-wide and strategy-level risk governance parameters."""

    max_risk_per_trade_pct: float = Field(
        default=0.015,
        ge=0.001,
        le=0.05,
        description="Maximum capital risked on a single trade (typically 1-2%)",
    )
    max_single_position_pct: float = Field(
        default=0.25,
        ge=0.05,
        le=0.50,
        description="Maximum allocation to a single symbol as fraction of equity",
    )
    max_open_positions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent open positions",
    )
    max_portfolio_leverage: float = Field(
        default=1.0,
        ge=1.0,
        le=5.0,
        description="Maximum total gross leverage allowed across portfolio",
    )
    daily_drawdown_limit_pct: float = Field(
        default=0.04,
        ge=0.01,
        le=0.10,
        description="Daily loss percentage from session open equity that halts new trading",
    )
    max_drawdown_limit_pct: float = Field(
        default=0.12,
        ge=0.03,
        le=0.25,
        description="Peak-to-trough overall drawdown percentage that triggers system trading halt",
    )
    min_risk_reward_ratio: float = Field(
        default=1.5,
        ge=1.0,
        description="Minimum acceptable risk/reward ratio for trade entry",
    )
