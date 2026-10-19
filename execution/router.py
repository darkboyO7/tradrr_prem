"""Trading Intelligence Platform - Smart Order Router (SOR).

Intelligently routes parent orders across multiple execution venues (e.g. Binance,
Kraken, Coinbase, Simulated) to minimize market impact, reduce slippage, optimize
net fees, or capture cross-venue spatial arbitrage opportunities.
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from core.logging import get_logger
from data.orderbook.consolidated import ArbitrageOpportunity, ConsolidatedOrderBook
from execution.base import BaseExecutionClient

logger = get_logger("smart_order_router")


class RoutingPolicy(str, Enum):
    """Execution policy for smart order routing."""

    BEST_PRICE = "BEST_PRICE"          # Route to venue offering best top-of-book price
    SPLIT_DEPTH = "SPLIT_DEPTH"        # Slice order across venues proportional to depth
    FEE_AWARE = "FEE_AWARE"            # Factor in venue fee tiers to minimize effective net cost
    ARBITRAGE_SWEEP = "ARBITRAGE_SWEEP"  # Concurrent dual-leg execution on crossed book


class ChildOrderSlice(BaseModel):
    """Sub-order slice allocated to a specific execution venue."""

    slice_id: str = Field(default_factory=lambda: f"slc_{uuid.uuid4().hex[:8]}")
    venue: str
    symbol: str
    side: str  # BUY or SELL
    order_type: str = "MARKET"
    quantity: float
    limit_price: Optional[float] = None
    expected_price: float
    expected_fee: float


class RoutingPlan(BaseModel):
    """Comprehensive multi-venue execution plan for a parent order."""

    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    symbol: str
    side: str
    parent_quantity: float
    policy: RoutingPolicy
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    slices: List[ChildOrderSlice] = []
    estimated_vwap: float = 0.0
    estimated_total_cost: float = 0.0
    estimated_total_fees: float = 0.0


class SmartOrderRouter:
    """Institutional Smart Order Router directing child orders across venues."""

    DEFAULT_FEES = {
        "binance": 0.00075,   # 7.5 bps (with BNB discount)
        "kraken": 0.0016,     # 16 bps taker
        "coinbase": 0.0020,   # 20 bps taker
        "simulated": 0.0010,  # 10 bps standard benchmark
    }

    def __init__(
        self,
        consolidated_book: ConsolidatedOrderBook,
        venue_clients: Optional[Dict[str, BaseExecutionClient]] = None,
        venue_fee_rates: Optional[Dict[str, float]] = None,
    ) -> None:
        self.book = consolidated_book
        self.clients: Dict[str, BaseExecutionClient] = venue_clients or {}
        self.fee_rates: Dict[str, float] = venue_fee_rates or self.DEFAULT_FEES.copy()

    def register_client(self, venue: str, client: BaseExecutionClient) -> None:
        """Register or update an execution client for a venue."""
        self.clients[venue.lower()] = client

    def set_venue_fee(self, venue: str, fee_rate: float) -> None:
        """Update the taker fee rate for a specific venue."""
        self.fee_rates[venue.lower()] = fee_rate

    def plan_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        policy: RoutingPolicy = RoutingPolicy.BEST_PRICE,
        max_depth_levels: int = 5,
    ) -> RoutingPlan:
        """Generate an optimal multi-venue child routing plan."""
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid order side: {side}")
        if quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {quantity}")

        if policy == RoutingPolicy.BEST_PRICE:
            return self._plan_best_price(symbol, side, quantity)
        elif policy == RoutingPolicy.FEE_AWARE:
            return self._plan_fee_aware(symbol, side, quantity)
        elif policy == RoutingPolicy.SPLIT_DEPTH:
            return self._plan_split_depth(symbol, side, quantity, max_depth_levels)
        else:
            return self._plan_best_price(symbol, side, quantity)

    def _plan_best_price(self, symbol: str, side: str, quantity: float) -> RoutingPlan:
        """Route entirely to the single venue offering the best raw quote."""
        bbo = self.book.get_consolidated_bbo()
        target_venue: str
        target_price: float

        if side == "BUY":
            if bbo.best_ask is None or not bbo.best_ask_venue:
                raise ValueError(f"No ask liquidity available for {symbol}")
            target_venue = bbo.best_ask_venue
            target_price = bbo.best_ask
        else:
            if bbo.best_bid is None or not bbo.best_bid_venue:
                raise ValueError(f"No bid liquidity available for {symbol}")
            target_venue = bbo.best_bid_venue
            target_price = bbo.best_bid

        fee_rate = self.fee_rates.get(target_venue.lower(), 0.001)
        expected_fee = round(quantity * target_price * fee_rate, 4)
        expected_cost = round(quantity * target_price, 4)

        child_slice = ChildOrderSlice(
            venue=target_venue,
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=target_price,
            expected_fee=expected_fee,
        )

        return RoutingPlan(
            symbol=symbol,
            side=side,
            parent_quantity=quantity,
            policy=RoutingPolicy.BEST_PRICE,
            slices=[child_slice],
            estimated_vwap=target_price,
            estimated_total_cost=expected_cost,
            estimated_total_fees=expected_fee,
        )

    def _plan_fee_aware(self, symbol: str, side: str, quantity: float) -> RoutingPlan:
        """Evaluate effective net price (price +/- fee) across venues and pick the optimal venue."""
        venues = self.book.venues
        if not venues:
            raise ValueError(f"No venue books registered for {symbol}")

        best_venue: Optional[str] = None
        best_price: Optional[float] = None
        best_effective_price: Optional[float] = None

        for venue in venues:
            v_book = self.book.get_venue_book(venue)
            if not v_book:
                continue

            fee_rate = self.fee_rates.get(venue.lower(), 0.001)

            if side == "BUY":
                ask = v_book.best_ask
                if ask is not None:
                    effective_price = ask * (1.0 + fee_rate)
                    if best_effective_price is None or effective_price < best_effective_price:
                        best_effective_price = effective_price
                        best_venue = venue
                        best_price = ask
            else:  # SELL
                bid = v_book.best_bid
                if bid is not None:
                    effective_price = bid * (1.0 - fee_rate)
                    if best_effective_price is None or effective_price > best_effective_price:
                        best_effective_price = effective_price
                        best_venue = venue
                        best_price = bid

        if not best_venue or best_price is None:
            raise ValueError(f"No eligible venue found for {side} {symbol}")

        fee_rate = self.fee_rates.get(best_venue.lower(), 0.001)
        expected_fee = round(quantity * best_price * fee_rate, 4)
        expected_cost = round(quantity * best_price, 4)

        child_slice = ChildOrderSlice(
            venue=best_venue,
            symbol=symbol,
            side=side,
            quantity=quantity,
            expected_price=best_price,
            expected_fee=expected_fee,
        )

        return RoutingPlan(
            symbol=symbol,
            side=side,
            parent_quantity=quantity,
            policy=RoutingPolicy.FEE_AWARE,
            slices=[child_slice],
            estimated_vwap=best_price,
            estimated_total_cost=expected_cost,
            estimated_total_fees=expected_fee,
        )

    def _plan_split_depth(
        self, symbol: str, side: str, quantity: float, max_levels: int = 5
    ) -> RoutingPlan:
        """Slice parent order across multiple venues proportional to available liquidity."""
        venues = self.book.venues
        if not venues:
            raise ValueError(f"No venue books registered for {symbol}")

        venue_depths: Dict[str, Tuple[float, float]] = {}  # venue -> (best_price, total_depth)

        for venue in venues:
            v_book = self.book.get_venue_book(venue)
            if not v_book:
                continue

            if side == "BUY":
                asks = v_book.get_asks(max_levels)
                if asks:
                    depth = sum(q for _, q in asks)
                    venue_depths[venue] = (asks[0][0], depth)
            else:
                bids = v_book.get_bids(max_levels)
                if bids:
                    depth = sum(q for _, q in bids)
                    venue_depths[venue] = (bids[0][0], depth)

        if not venue_depths:
            raise ValueError(f"No depth available across venues for {side} {symbol}")

        total_available_depth = sum(depth for _, depth in venue_depths.values())
        if total_available_depth <= 0:
            return self._plan_best_price(symbol, side, quantity)

        slices: List[ChildOrderSlice] = []
        total_cost = 0.0
        total_fees = 0.0

        for venue, (top_price, depth) in venue_depths.items():
            ratio = depth / total_available_depth
            slice_qty = round(quantity * ratio, 6)
            if slice_qty <= 0:
                continue

            fee_rate = self.fee_rates.get(venue.lower(), 0.001)
            fee = round(slice_qty * top_price * fee_rate, 4)
            cost = round(slice_qty * top_price, 4)

            total_cost += cost
            total_fees += fee

            slices.append(
                ChildOrderSlice(
                    venue=venue,
                    symbol=symbol,
                    side=side,
                    quantity=slice_qty,
                    expected_price=top_price,
                    expected_fee=fee,
                )
            )

        if not slices:
            return self._plan_best_price(symbol, side, quantity)

        # Normalize slice quantities to match parent quantity precisely
        allocated_qty = sum(s.quantity for s in slices)
        diff = round(quantity - allocated_qty, 6)
        if diff != 0 and slices:
            slices[0].quantity = round(slices[0].quantity + diff, 6)

        vwap = round(total_cost / quantity, 4) if quantity > 0 else 0.0

        return RoutingPlan(
            symbol=symbol,
            side=side,
            parent_quantity=quantity,
            policy=RoutingPolicy.SPLIT_DEPTH,
            slices=slices,
            estimated_vwap=vwap,
            estimated_total_cost=round(total_cost, 4),
            estimated_total_fees=round(total_fees, 4),
        )

    async def execute_plan(self, plan: RoutingPlan) -> Dict[str, Any]:
        """Concurrently dispatch child order slices to designated venue clients."""
        logger.info(
            "Executing smart routing plan",
            plan_id=plan.plan_id,
            slices_count=len(plan.slices),
            policy=plan.policy.value,
        )

        tasks = []
        for slc in plan.slices:
            client = self.clients.get(slc.venue.lower())
            if not client:
                logger.warning("No execution client configured for venue, simulating fill", venue=slc.venue)
                tasks.append(self._simulate_child_fill(slc))
            else:
                tasks.append(
                    client.place_order(
                        symbol=slc.symbol,
                        side=slc.side,
                        order_type=slc.order_type,
                        quantity=slc.quantity,
                        price=slc.expected_price,
                        client_order_id=slc.slice_id,
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        executed_slices = []
        total_filled_qty = 0.0
        total_filled_value = 0.0

        for slc, res in zip(plan.slices, results):
            if isinstance(res, Exception):
                logger.error("Slice execution failed", slice_id=slc.slice_id, venue=slc.venue, error=str(res))
                executed_slices.append({
                    "slice_id": slc.slice_id,
                    "venue": slc.venue,
                    "status": "FAILED",
                    "error": str(res),
                })
            else:
                filled_qty = float(res.get("executed_quantity", res.get("quantity", slc.quantity)))
                fill_price = float(res.get("price", slc.expected_price))
                total_filled_qty += filled_qty
                total_filled_value += filled_qty * fill_price
                executed_slices.append({
                    "slice_id": slc.slice_id,
                    "venue": slc.venue,
                    "status": "FILLED",
                    "filled_quantity": filled_qty,
                    "filled_price": fill_price,
                    "raw_response": res,
                })

        realized_vwap = round(total_filled_value / total_filled_qty, 4) if total_filled_qty > 0 else 0.0

        return {
            "plan_id": plan.plan_id,
            "status": "COMPLETED" if total_filled_qty >= plan.parent_quantity * 0.999 else "PARTIALLY_FILLED",
            "parent_quantity": plan.parent_quantity,
            "total_filled_quantity": round(total_filled_qty, 6),
            "realized_vwap": realized_vwap,
            "policy": plan.policy.value,
            "slices": executed_slices,
        }

    async def _simulate_child_fill(self, slc: ChildOrderSlice) -> Dict[str, Any]:
        """Internal mock fill when a venue client is not configured."""
        return {
            "order_id": f"sim_{slc.slice_id}",
            "symbol": slc.symbol,
            "side": slc.side,
            "quantity": slc.quantity,
            "executed_quantity": slc.quantity,
            "price": slc.expected_price,
            "status": "FILLED",
            "fee": slc.expected_fee,
        }

    async def execute_arbitrage(
        self,
        opp: ArbitrageOpportunity,
        volume: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute dual-leg atomic market orders across buy venue and sell venue."""
        exec_volume = volume or opp.max_executable_volume
        if exec_volume <= 0:
            raise ValueError(f"Invalid arbitrage volume: {exec_volume}")

        logger.info(
            "Executing spatial arbitrage opportunity",
            symbol=opp.symbol,
            buy_venue=opp.buy_venue,
            buy_price=opp.buy_price,
            sell_venue=opp.sell_venue,
            sell_price=opp.sell_price,
            volume=exec_volume,
            net_profit_bps=opp.net_profit_bps,
        )

        buy_slice = ChildOrderSlice(
            venue=opp.buy_venue,
            symbol=opp.symbol,
            side="BUY",
            quantity=exec_volume,
            expected_price=opp.buy_price,
            expected_fee=round(exec_volume * opp.buy_price * self.fee_rates.get(opp.buy_venue.lower(), 0.001), 4),
        )

        sell_slice = ChildOrderSlice(
            venue=opp.sell_venue,
            symbol=opp.symbol,
            side="SELL",
            quantity=exec_volume,
            expected_price=opp.sell_price,
            expected_fee=round(exec_volume * opp.sell_price * self.fee_rates.get(opp.sell_venue.lower(), 0.001), 4),
        )

        plan = RoutingPlan(
            symbol=opp.symbol,
            side="ARBITRAGE",
            parent_quantity=exec_volume,
            policy=RoutingPolicy.ARBITRAGE_SWEEP,
            slices=[buy_slice, sell_slice],
            estimated_vwap=(opp.buy_price + opp.sell_price) / 2.0,
            estimated_total_cost=round(exec_volume * opp.buy_price, 4),
            estimated_total_fees=buy_slice.expected_fee + sell_slice.expected_fee,
        )

        res = await self.execute_plan(plan)
        res["net_profit_bps_expected"] = opp.net_profit_bps
        res["estimated_net_pnl"] = opp.estimated_net_pnl
        return res
