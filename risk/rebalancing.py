"""Trading Intelligence Platform - Optimal Portfolio Rebalancer.

Calculates asset weight drift against targets, filters micro-turnover using deadbands,
estimates execution drag via Almgren-Chriss market impact, and generates optimal
sequenced rebalancing orders (sells prior to buys).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from core.logging import get_logger
from execution.cost_model import AlmgrenChrissCostModel, ExecutionCostEstimate

logger = get_logger("rebalancer")


class RebalanceOrder(BaseModel):
    """Specific asset order required to realign portfolio with target weights."""

    order_id: str = Field(default_factory=lambda: f"reb_{uuid.uuid4().hex[:8]}")
    symbol: str
    side: str  # BUY or SELL
    target_units: float
    current_units: float
    order_quantity: float
    current_price: float
    notional_value_usd: float
    drift_pct: float
    cost_estimate: ExecutionCostEstimate


class RebalancePlan(BaseModel):
    """Institutional portfolio rebalancing schedule with market impact estimates."""

    plan_id: str = Field(default_factory=lambda: f"rebal_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_equity: float
    initial_cash: float
    target_cash: float
    estimated_turnover_usd: float
    estimated_total_cost_usd: float
    estimated_cost_bps: float
    orders: List[RebalanceOrder] = []


class PortfolioRebalancer:
    """Calculates optimal rebalancing orders while minimizing turnover churn and impact drag."""

    def __init__(
        self,
        deadband_drift_pct: float = 0.02,  # 2% drift threshold to prevent churn
        cost_model: Optional[AlmgrenChrissCostModel] = None,
    ) -> None:
        self.deadband_drift_pct = deadband_drift_pct
        self.cost_model = cost_model or AlmgrenChrissCostModel()

    def generate_plan(
        self,
        current_holdings: Dict[str, float],
        current_cash: float,
        current_prices: Dict[str, float],
        target_weights: Dict[str, float],
        average_daily_volumes: Optional[Dict[str, float]] = None,
    ) -> RebalancePlan:
        """Construct sequenced rebalancing plan prioritizing capital-freeing sells."""
        adv_map = average_daily_volumes or {}

        # 1. Calculate Current Portfolio Equity
        holdings_value = sum(
            current_holdings.get(sym, 0.0) * current_prices[sym]
            for sym in current_holdings.keys()
            if sym in current_prices
        )
        total_equity = current_cash + holdings_value

        if total_equity <= 0:
            raise ValueError(f"Total portfolio equity must be positive, got {total_equity}")

        # All relevant symbols (current holdings + target allocations)
        all_symbols = set(current_holdings.keys()).union(target_weights.keys())

        sell_orders: List[RebalanceOrder] = []
        buy_orders: List[RebalanceOrder] = []
        total_turnover = 0.0
        total_costs = 0.0

        for sym in all_symbols:
            price = current_prices.get(sym)
            if price is None or price <= 0:
                logger.warning("Missing valid price for asset during rebalancing, skipping", symbol=sym)
                continue

            curr_qty = current_holdings.get(sym, 0.0)
            curr_val = curr_qty * price
            curr_weight = curr_val / total_equity

            targ_weight = target_weights.get(sym, 0.0)
            drift = targ_weight - curr_weight

            # Deadband check: ignore minor fluctuations within tolerance
            if abs(drift) < self.deadband_drift_pct:
                continue

            # Target allocation in USD and units
            targ_val = targ_weight * total_equity
            targ_qty = targ_val / price
            delta_qty = targ_qty - curr_qty

            side = "BUY" if delta_qty > 0 else "SELL"
            trade_qty = abs(delta_qty)
            trade_notional = trade_qty * price

            adv = adv_map.get(sym, 5000.0)
            cost_est = self.cost_model.estimate_cost(
                symbol=sym,
                side=side,
                quantity=trade_qty,
                current_price=price,
                average_daily_volume=adv,
            )

            order = RebalanceOrder(
                symbol=sym,
                side=side,
                target_units=round(targ_qty, 6),
                current_units=round(curr_qty, 6),
                order_quantity=round(trade_qty, 6),
                current_price=price,
                notional_value_usd=round(trade_notional, 2),
                drift_pct=round(drift * 100.0, 2),
                cost_estimate=cost_est,
            )

            total_turnover += trade_notional
            total_costs += cost_est.total_cost_usd

            if side == "SELL":
                sell_orders.append(order)
            else:
                buy_orders.append(order)

        # Sequence SELLS first to release cash, followed by BUYS
        sequenced_orders = sell_orders + buy_orders

        cost_bps = (total_costs / total_turnover) * 10_000.0 if total_turnover > 0 else 0.0
        target_cash_val = total_equity * max(0.0, 1.0 - sum(target_weights.values()))

        logger.info(
            "Portfolio rebalance plan constructed",
            total_equity=round(total_equity, 2),
            orders_count=len(sequenced_orders),
            turnover_usd=round(total_turnover, 2),
            total_cost_usd=round(total_costs, 2),
        )

        return RebalancePlan(
            total_equity=round(total_equity, 2),
            initial_cash=round(current_cash, 2),
            target_cash=round(target_cash_val, 2),
            estimated_turnover_usd=round(total_turnover, 2),
            estimated_total_cost_usd=round(total_costs, 2),
            estimated_cost_bps=round(cost_bps, 2),
            orders=sequenced_orders,
        )
