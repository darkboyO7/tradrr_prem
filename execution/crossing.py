"""Trading Intelligence Platform - Internal Matching & Crossing Engine.

Implements an institutional crossing pool / internal matching network that crosses
internal orders at mid-price before routing residual flow to external venues,
eliminating exchange taker fees and bid-ask spread slippage.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from core.logging import get_logger
from execution.router import RoutingPlan, RoutingPolicy, SmartOrderRouter

logger = get_logger("crossing_engine")


class CrossingPriority(str, Enum):
    """Order crossing prioritization algorithm."""

    PRICE_TIME = "PRICE_TIME"  # Best limit price protection first, then FIFO
    PRO_RATA = "PRO_RATA"      # Proportional allocation by order size


class CrossingTimeInForce(str, Enum):
    """Crossing order time-in-force instructions."""

    IOC = "IOC"          # Immediate-or-Cancel: match available resting flow, cancel or route remainder
    FOK = "FOK"          # Fill-or-Kill: match full size at midpoint or abort completely
    RESTING = "RESTING"  # Rest unfilled remainder in internal crossing pool


class CrossingSide(str, Enum):
    """Order transaction side."""

    BUY = "BUY"
    SELL = "SELL"


class InternalCrossingOrder(BaseModel):
    """Order submitted to internal midpoint crossing network."""

    order_id: str = Field(default_factory=lambda: f"cx_{uuid.uuid4().hex[:8]}")
    symbol: str
    side: CrossingSide
    quantity: float
    remaining_quantity: float
    limit_price: Optional[float] = None
    time_in_force: CrossingTimeInForce = CrossingTimeInForce.IOC
    strategy_id: str = "DEFAULT"
    min_fill_quantity: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CrossingFill(BaseModel):
    """Executed internal crossing match record."""

    fill_id: str = Field(default_factory=lambda: f"cxf_{uuid.uuid4().hex[:8]}")
    buy_order_id: str
    sell_order_id: str
    symbol: str
    match_price: float
    quantity: float
    notional_usd: float
    spread_savings_usd: float
    fee_savings_usd: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CrossingResult(BaseModel):
    """Outcome of internal crossing execution attempt."""

    incoming_order_id: str
    symbol: str
    side: CrossingSide
    requested_quantity: float
    filled_quantity: float
    remaining_unfilled_quantity: float
    average_match_price: float
    fills: List[CrossingFill] = []
    total_spread_savings_usd: float
    total_fee_savings_usd: float
    residual_routed_externally: bool = False
    external_routing_plan: Optional[RoutingPlan] = None


class CrossingPoolMetrics(BaseModel):
    """Institutional dark pool depth, volume, and cumulative fee savings."""

    symbol: str
    resting_buy_volume: float
    resting_sell_volume: float
    resting_buy_count: int
    resting_sell_count: int
    cumulative_crossed_volume: float
    cumulative_crossed_notional_usd: float
    cumulative_spread_savings_usd: float
    cumulative_fee_savings_usd: float
    cumulative_total_savings_usd: float


class InternalCrossingEngine:
    """Matches institutional order flow internally at reference mid-price."""

    def __init__(
        self,
        default_taker_fee_rate: float = 0.0005,  # 5 bps
        priority: CrossingPriority = CrossingPriority.PRICE_TIME,
        router: Optional[SmartOrderRouter] = None,
    ) -> None:
        self.default_taker_fee_rate = default_taker_fee_rate
        self.priority = priority
        self.router = router

        # Resting internal books: {symbol: [InternalCrossingOrder]}
        self._resting_buys: Dict[str, List[InternalCrossingOrder]] = {}
        self._resting_sells: Dict[str, List[InternalCrossingOrder]] = {}

        # Telemetry metrics: {symbol: dict}
        self._metrics: Dict[str, Dict[str, float]] = {}

    def _ensure_symbol_storage(self, symbol: str) -> None:
        if symbol not in self._resting_buys:
            self._resting_buys[symbol] = []
        if symbol not in self._resting_sells:
            self._resting_sells[symbol] = []
        if symbol not in self._metrics:
            self._metrics[symbol] = {
                "crossed_volume": 0.0,
                "crossed_notional_usd": 0.0,
                "spread_savings_usd": 0.0,
                "fee_savings_usd": 0.0,
            }

    def cross_order(
        self,
        order: InternalCrossingOrder,
        current_bid: float,
        current_ask: float,
        route_residual_external: bool = False,
    ) -> CrossingResult:
        """Execute order against internal crossing pool at reference midpoint."""
        self._ensure_symbol_storage(order.symbol)
        sym = order.symbol
        mid_price = (current_bid + current_ask) / 2.0
        half_spread = (current_ask - current_bid) / 2.0

        is_buy = order.side == CrossingSide.BUY
        contra_list = self._resting_sells[sym] if is_buy else self._resting_buys[sym]

        # Verify limit price admissibility against midpoint
        if order.limit_price is not None:
            if is_buy and mid_price > order.limit_price:
                return CrossingResult(
                    incoming_order_id=order.order_id,
                    symbol=sym,
                    side=order.side,
                    requested_quantity=order.quantity,
                    filled_quantity=0.0,
                    remaining_unfilled_quantity=order.quantity,
                    average_match_price=0.0,
                    fills=[],
                    total_spread_savings_usd=0.0,
                    total_fee_savings_usd=0.0,
                    residual_routed_externally=False,
                )
            if not is_buy and mid_price < order.limit_price:
                return CrossingResult(
                    incoming_order_id=order.order_id,
                    symbol=sym,
                    side=order.side,
                    requested_quantity=order.quantity,
                    filled_quantity=0.0,
                    remaining_unfilled_quantity=order.quantity,
                    average_match_price=0.0,
                    fills=[],
                    total_spread_savings_usd=0.0,
                    total_fee_savings_usd=0.0,
                    residual_routed_externally=False,
                )

        # Pre-check available contra liquidity for FOK orders
        available_contra_qty = 0.0
        eligible_contra_orders = []
        for contra in contra_list:
            if contra.limit_price is not None:
                if is_buy and mid_price < contra.limit_price:
                    continue  # Seller limit is higher than midpoint
                if not is_buy and mid_price > contra.limit_price:
                    continue  # Buyer limit is lower than midpoint
            available_contra_qty += contra.remaining_quantity
            eligible_contra_orders.append(contra)

        if order.time_in_force == CrossingTimeInForce.FOK and available_contra_qty < order.quantity:
            logger.info(f"FOK order {order.order_id} failed: required {order.quantity}, available {available_contra_qty}")
            return CrossingResult(
                incoming_order_id=order.order_id,
                symbol=sym,
                side=order.side,
                requested_quantity=order.quantity,
                filled_quantity=0.0,
                remaining_unfilled_quantity=order.quantity,
                average_match_price=0.0,
                fills=[],
                total_spread_savings_usd=0.0,
                total_fee_savings_usd=0.0,
                residual_routed_externally=False,
            )

        # Match incoming order against eligible resting contra orders
        fills: List[CrossingFill] = []
        filled_qty = 0.0
        remaining_qty = order.quantity

        for contra in eligible_contra_orders:
            if remaining_qty <= 1e-7:
                break

            match_qty = min(remaining_qty, contra.remaining_quantity)
            if match_qty <= 0:
                continue

            notional = match_qty * mid_price
            # Spread savings: half-spread saved for buyer + half-spread saved for seller = full spread saved
            spread_saving = (2.0 * half_spread) * match_qty
            # Fee savings: 2 * taker fee rate avoided
            fee_saving = 2.0 * self.default_taker_fee_rate * notional

            fill = CrossingFill(
                buy_order_id=order.order_id if is_buy else contra.order_id,
                sell_order_id=contra.order_id if is_buy else order.order_id,
                symbol=sym,
                match_price=mid_price,
                quantity=round(match_qty, 6),
                notional_usd=round(notional, 2),
                spread_savings_usd=round(spread_saving, 2),
                fee_savings_usd=round(fee_saving, 2),
            )
            fills.append(fill)

            filled_qty += match_qty
            remaining_qty -= match_qty
            contra.remaining_quantity -= match_qty

            # Update metrics
            self._metrics[sym]["crossed_volume"] += match_qty
            self._metrics[sym]["crossed_notional_usd"] += notional
            self._metrics[sym]["spread_savings_usd"] += spread_saving
            self._metrics[sym]["fee_savings_usd"] += fee_saving

        # Remove completely filled contra orders from resting pool
        if is_buy:
            self._resting_sells[sym] = [c for c in self._resting_sells[sym] if c.remaining_quantity > 1e-7]
        else:
            self._resting_buys[sym] = [c for c in self._resting_buys[sym] if c.remaining_quantity > 1e-7]

        # Handle unfilled remainder
        routed_external = False
        external_plan = None

        if remaining_qty > 1e-7:
            if order.time_in_force == CrossingTimeInForce.RESTING:
                order.remaining_quantity = remaining_qty
                if is_buy:
                    self._resting_buys[sym].append(order)
                else:
                    self._resting_sells[sym].append(order)
                logger.info(f"Resting order remainder {order.order_id}: {remaining_qty:.4f} {sym} posted to internal book.")
            elif route_residual_external and self.router is not None:
                external_plan = self.router.plan_order(
                    symbol=sym,
                    side=order.side.value,
                    quantity=remaining_qty,
                    policy=RoutingPolicy.BEST_PRICE,
                )
                routed_external = True
                logger.info(f"Routed residual {remaining_qty:.4f} {sym} externally via SmartOrderRouter.")

        avg_price = mid_price if filled_qty > 0 else 0.0
        total_spread_savings = sum(f.spread_savings_usd for f in fills)
        total_fee_savings = sum(f.fee_savings_usd for f in fills)

        return CrossingResult(
            incoming_order_id=order.order_id,
            symbol=sym,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=round(filled_qty, 6),
            remaining_unfilled_quantity=round(remaining_qty, 6),
            average_match_price=round(avg_price, 4),
            fills=fills,
            total_spread_savings_usd=round(total_spread_savings, 2),
            total_fee_savings_usd=round(total_fee_savings, 2),
            residual_routed_externally=routed_external,
            external_routing_plan=external_plan,
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a resting order in the internal crossing book."""
        self._ensure_symbol_storage(symbol)
        buys = self._resting_buys[symbol]
        sells = self._resting_sells[symbol]

        for i, o in enumerate(buys):
            if o.order_id == order_id:
                buys.pop(i)
                return True

        for i, o in enumerate(sells):
            if o.order_id == order_id:
                sells.pop(i)
                return True

        return False

    def get_metrics(self, symbol: str) -> CrossingPoolMetrics:
        """Retrieve operational statistics and cumulative fee/spread savings."""
        self._ensure_symbol_storage(symbol)
        buys = self._resting_buys[symbol]
        sells = self._resting_sells[symbol]
        m = self._metrics[symbol]

        buy_vol = sum(o.remaining_quantity for o in buys)
        sell_vol = sum(o.remaining_quantity for o in sells)
        tot_savings = m["spread_savings_usd"] + m["fee_savings_usd"]

        return CrossingPoolMetrics(
            symbol=symbol,
            resting_buy_volume=round(buy_vol, 6),
            resting_sell_volume=round(sell_vol, 6),
            resting_buy_count=len(buys),
            resting_sell_count=len(sells),
            cumulative_crossed_volume=round(m["crossed_volume"], 6),
            cumulative_crossed_notional_usd=round(m["crossed_notional_usd"], 2),
            cumulative_spread_savings_usd=round(m["spread_savings_usd"], 2),
            cumulative_fee_savings_usd=round(m["fee_savings_usd"], 2),
            cumulative_total_savings_usd=round(tot_savings, 2),
        )

    def get_resting_book(self, symbol: str) -> Dict[str, Any]:
        """Inspect active internal resting buy and sell orders."""
        self._ensure_symbol_storage(symbol)
        return {
            "symbol": symbol,
            "resting_buys": [o.model_dump() for o in self._resting_buys[symbol]],
            "resting_sells": [o.model_dump() for o in self._resting_sells[symbol]],
        }
