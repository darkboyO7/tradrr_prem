"""Trading Intelligence Platform - Execution Cost & Market Impact Model.

Implements the institutional Almgren-Chriss market impact framework to estimate
half-spread drag, temporary price impact (square-root law), permanent price impact,
exchange taker fees, and total expected implementation shortfall.
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("cost_model")


class ExecutionCostEstimate(BaseModel):
    """Detailed breakdown of estimated transaction costs and market impact."""

    symbol: str
    side: str
    quantity: float
    current_price: float
    notional_value_usd: float

    # Component costs in USD
    half_spread_cost_usd: float
    temporary_impact_usd: float
    permanent_impact_usd: float
    venue_fees_usd: float
    total_cost_usd: float

    # Costs expressed in Basis Points (bps)
    half_spread_bps: float
    temporary_impact_bps: float
    permanent_impact_bps: float
    venue_fees_bps: float
    total_shortfall_bps: float

    # Market participation metrics
    participation_rate: float = Field(
        description="Trade quantity relative to expected volume during execution horizon"
    )
    recommended_strategy: str = Field(
        description="Execution directive: IMMEDIATE_SWEEP, TWAP_SLICING, or VWAP_ALGORITHM"
    )
    suggested_slices_count: int


class AlmgrenChrissCostModel:
    """Quantitative transaction cost and market impact estimator."""

    def __init__(
        self,
        temp_impact_coeff: float = 0.15,   # Eta (temporary impact parameter)
        perm_impact_coeff: float = 0.10,   # Gamma (permanent impact parameter)
        default_taker_fee: float = 0.0010, # 10 bps default taker fee
    ) -> None:
        self.eta = temp_impact_coeff
        self.gamma = perm_impact_coeff
        self.default_taker_fee = default_taker_fee

    def estimate_cost(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        bid_ask_spread: Optional[float] = None,
        average_daily_volume: float = 5000.0,
        daily_volatility: float = 0.03,  # 3% daily volatility
        execution_horizon_hours: float = 0.25,  # 15 minutes default
        venue_fee_rate: Optional[float] = None,
    ) -> ExecutionCostEstimate:
        """Estimate execution shortfall and market impact for a proposed order.
        
        Args:
            symbol: Ticker symbol (e.g. "BTC-USD").
            side: "BUY" or "SELL".
            quantity: Order size in base asset units.
            current_price: Current market mid-price.
            bid_ask_spread: Current top-of-book spread (defaults to 1.5 bps if None).
            average_daily_volume: 24h trading volume in base asset units.
            daily_volatility: Asset daily volatility (standard deviation of log returns).
            execution_horizon_hours: Planned duration of the execution window in hours.
            venue_fee_rate: Taker fee rate (e.g. 0.00075 for Binance).
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if current_price <= 0:
            raise ValueError(f"Price must be positive, got {current_price}")

        side = side.upper()
        notional_usd = quantity * current_price
        fee_rate = venue_fee_rate if venue_fee_rate is not None else self.default_taker_fee

        # 1. Half-Spread Drag
        # Default spread = 1.5 bps if not provided
        spread = bid_ask_spread if bid_ask_spread is not None else (current_price * 0.00015)
        half_spread_usd = 0.5 * spread * quantity
        half_spread_bps = (half_spread_usd / notional_usd) * 10_000.0

        # 2. Participation Rate over Execution Horizon
        # Fraction of daily volume expected during the horizon window
        horizon_fraction = max(execution_horizon_hours / 24.0, 1.0 / 1440.0)  # at least 1 min
        horizon_volume = max(average_daily_volume * horizon_fraction, 1e-6)
        participation_rate = quantity / horizon_volume

        # 3. Temporary Market Impact (Square Root Law)
        # I_temp = eta * sqrt(Q / V_horizon) * sigma_daily * Price * Q
        sqrt_participation = math.sqrt(min(participation_rate, 5.0))  # bound extreme illiquidity
        temp_impact_pct = self.eta * sqrt_participation * daily_volatility
        temp_impact_usd = temp_impact_pct * notional_usd
        temp_impact_bps = temp_impact_pct * 10_000.0

        # 4. Permanent Market Impact (Linear Law)
        # I_perm = gamma * (Q / ADV) * sigma_daily * Price * Q
        adv_participation = quantity / max(average_daily_volume, 1e-6)
        perm_impact_pct = self.gamma * adv_participation * daily_volatility
        perm_impact_usd = perm_impact_pct * notional_usd
        perm_impact_bps = perm_impact_pct * 10_000.0

        # 5. Venue Fees
        fees_usd = notional_usd * fee_rate
        fees_bps = fee_rate * 10_000.0

        # 6. Total Shortfall
        total_cost_usd = half_spread_usd + temp_impact_usd + perm_impact_usd + fees_usd
        total_shortfall_bps = (total_cost_usd / notional_usd) * 10_000.0

        # 7. Recommended Execution Strategy based on participation rate
        if participation_rate < 0.01:
            rec_strategy = "IMMEDIATE_SWEEP"
            slices = 1
        elif participation_rate <= 0.05:
            rec_strategy = "TWAP_SLICING"
            slices = max(3, int(execution_horizon_hours * 12))  # e.g. every 5 minutes
        else:
            rec_strategy = "VWAP_ALGORITHM"
            slices = max(6, int(execution_horizon_hours * 24))

        return ExecutionCostEstimate(
            symbol=symbol,
            side=side,
            quantity=round(quantity, 6),
            current_price=round(current_price, 4),
            notional_value_usd=round(notional_usd, 2),
            half_spread_cost_usd=round(half_spread_usd, 4),
            temporary_impact_usd=round(temp_impact_usd, 4),
            permanent_impact_usd=round(perm_impact_usd, 4),
            venue_fees_usd=round(fees_usd, 4),
            total_cost_usd=round(total_cost_usd, 4),
            half_spread_bps=round(half_spread_bps, 2),
            temporary_impact_bps=round(temp_impact_bps, 2),
            permanent_impact_bps=round(perm_impact_bps, 2),
            venue_fees_bps=round(fees_bps, 2),
            total_shortfall_bps=round(total_shortfall_bps, 2),
            participation_rate=round(participation_rate, 4),
            recommended_strategy=rec_strategy,
            suggested_slices_count=slices,
        )
