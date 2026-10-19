"""Trading Intelligence Platform - Cross-Asset Smart Hedging & Real-Time Greeks Rebalancer.

Monitors net portfolio Greek exposures (Delta, Gamma, Vega), enforces deadbands
to prevent excessive churn, estimates transaction costs and market impact via
the Almgren-Chriss framework, and synthesizes optimal spot/perpetual hedge orders.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from analysis.derivatives import DerivativePortfolioReport
from core.logging import get_logger
from execution.base import BaseExecutionClient
from execution.cost_model import AlmgrenChrissCostModel, ExecutionCostEstimate

logger = get_logger("smart_hedging")


class HedgeInstrument(str, Enum):
    """Institutional execution vehicle for derivative Greek hedging."""

    SPOT = "SPOT"
    PERPETUAL = "PERPETUAL"
    OPTION = "OPTION"


class HedgeTriggerType(str, Enum):
    """Classification of trigger event initiating a rebalancing / hedging sequence."""

    DELTA_BAND_BREACH = "DELTA_BAND_BREACH"
    GAMMA_EXPOSURE_LIMIT = "GAMMA_EXPOSURE_LIMIT"
    VEGA_RISK_LIMIT = "VEGA_RISK_LIMIT"
    PERIODIC_REBALANCE = "PERIODIC_REBALANCE"
    TARGET_ALIGNED = "TARGET_ALIGNED"


class HedgeOrder(BaseModel):
    """Individual rebalancing order for neutralizing portfolio Greek exposure."""

    order_id: str = Field(default_factory=lambda: f"hdg_{uuid.uuid4().hex[:8]}")
    symbol: str
    instrument_type: HedgeInstrument
    side: str  # "BUY" or "SELL"
    quantity: float
    target_price: float
    notional_usd: float
    execution_strategy: str  # "IMMEDIATE" or "TWAP" or "VWAP"
    estimated_cost: ExecutionCostEstimate
    projected_post_hedge_delta: float


class SmartHedgeConfig(BaseModel):
    """Configuration parameters and deadbands for automated Greek rebalancing."""

    delta_tolerance: float = 0.50          # Max absolute delta imbalance allowed before hedge
    gamma_max_dollar: float = 50000.0      # Max dollar gamma risk threshold
    vega_tolerance_1pct: float = 10000.0   # Max 1% vega exposure USD
    default_instrument: HedgeInstrument = HedgeInstrument.PERPETUAL
    twap_threshold_usd: float = 100000.0   # Notional threshold to route via TWAP algo
    twap_duration_minutes: int = 15
    twap_slices: int = 5


class SmartHedgePlan(BaseModel):
    """Complete institutional smart hedge decision and execution schedule."""

    plan_id: str = Field(default_factory=lambda: f"hplan_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trigger_reason: HedgeTriggerType
    pre_hedge_delta: float
    pre_hedge_gamma: float
    pre_hedge_vega_1pct: float
    projected_post_hedge_delta: float
    total_turnover_usd: float
    total_estimated_cost_usd: float
    hedging_required: bool
    orders: List[HedgeOrder] = []
    action_summary: str


class SmartHedgingEngine:
    """Evaluates Greek imbalances and generates cost-optimized hedging execution plans."""

    def __init__(
        self,
        config: Optional[SmartHedgeConfig] = None,
        cost_model: Optional[AlmgrenChrissCostModel] = None,
    ) -> None:
        self.config = config or SmartHedgeConfig()
        self.cost_model = cost_model or AlmgrenChrissCostModel()

    def generate_hedge_plan(
        self,
        portfolio_greeks: DerivativePortfolioReport,
        spot_symbol: str,
        spot_price: float,
        perp_symbol: Optional[str] = None,
        perp_price: Optional[float] = None,
        average_daily_volume: float = 10000.0,
        daily_volatility: float = 0.04,
    ) -> SmartHedgePlan:
        """Construct cost-optimized delta-neutral hedging plan based on Greek thresholds."""
        net_delta = portfolio_greeks.net_delta
        net_gamma = portfolio_greeks.net_gamma
        net_vega = portfolio_greeks.net_vega_1pct_usd
        abs_delta = abs(net_delta)

        # Check deadbands: if within tolerance and no limit breaches, no hedge needed
        if abs_delta <= self.config.delta_tolerance:
            logger.info(
                f"Portfolio delta ({net_delta:.4f}) within tolerance ({self.config.delta_tolerance}). No hedge required."
            )
            return SmartHedgePlan(
                trigger_reason=HedgeTriggerType.TARGET_ALIGNED,
                pre_hedge_delta=net_delta,
                pre_hedge_gamma=net_gamma,
                pre_hedge_vega_1pct=net_vega,
                projected_post_hedge_delta=net_delta,
                total_turnover_usd=0.0,
                total_estimated_cost_usd=0.0,
                hedging_required=False,
                orders=[],
                action_summary=f"Delta {net_delta:.4f} is within deadband tolerance +/-{self.config.delta_tolerance}.",
            )

        # Delta breach identified -> determine required hedge quantity
        hedge_shares = portfolio_greeks.delta_neutral_hedge_shares
        side = "BUY" if hedge_shares > 0 else "SELL"
        qty = abs(hedge_shares)

        # Select hedge instrument (perpetual or spot)
        use_perp = (self.config.default_instrument == HedgeInstrument.PERPETUAL) and (perp_symbol is not None)
        symbol = perp_symbol if use_perp else spot_symbol
        exec_price = (perp_price if use_perp and perp_price else spot_price)
        inst_type = HedgeInstrument.PERPETUAL if use_perp else HedgeInstrument.SPOT

        notional = qty * exec_price

        # Estimate execution drag and market impact via Almgren-Chriss
        cost_est = self.cost_model.estimate_cost(
            symbol=symbol,
            side=side,
            quantity=qty,
            current_price=exec_price,
            average_daily_volume=average_daily_volume,
            daily_volatility=daily_volatility,
        )

        # Determine strategy: TWAP slicing for large orders to minimize impact
        exec_strategy = "TWAP" if notional >= self.config.twap_threshold_usd else "IMMEDIATE"

        hedge_order = HedgeOrder(
            symbol=symbol,
            instrument_type=inst_type,
            side=side,
            quantity=round(qty, 6),
            target_price=exec_price,
            notional_usd=round(notional, 2),
            execution_strategy=exec_strategy,
            estimated_cost=cost_est,
            projected_post_hedge_delta=0.0,
        )

        summary = (
            f"Delta breach ({net_delta:.4f} > {self.config.delta_tolerance}). "
            f"Generated {side} {qty:.4f} {symbol} order ({exec_strategy}) to neutralize delta. "
            f"Est cost: ${cost_est.total_cost_usd:.2f} ({cost_est.total_shortfall_bps:.1f} bps)."
        )

        logger.info(f"Smart hedge plan synthesized: {summary}")

        return SmartHedgePlan(
            trigger_reason=HedgeTriggerType.DELTA_BAND_BREACH,
            pre_hedge_delta=net_delta,
            pre_hedge_gamma=net_gamma,
            pre_hedge_vega_1pct=net_vega,
            projected_post_hedge_delta=0.0,
            total_turnover_usd=round(notional, 2),
            total_estimated_cost_usd=round(cost_est.total_cost_usd, 2),
            hedging_required=True,
            orders=[hedge_order],
            action_summary=summary,
        )

    async def execute_hedge_plan(
        self,
        plan: SmartHedgePlan,
        client: BaseExecutionClient,
    ) -> List[Dict[str, Any]]:
        """Dispatch generated hedge orders to execution venue client."""
        results = []
        if not plan.hedging_required or not plan.orders:
            logger.info("No hedge orders to dispatch.")
            return results

        for order in plan.orders:
            logger.info(
                f"Dispatching smart hedge order: {order.side} {order.quantity} {order.symbol} @ {order.target_price}"
            )
            report = await client.place_order(
                symbol=order.symbol,
                side=order.side,
                order_type="MARKET",
                quantity=order.quantity,
                price=order.target_price,
            )
            results.append(report)

        return results
