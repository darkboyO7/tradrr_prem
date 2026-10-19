"""Cross-Venue Synthetic Smart Order Routing (SOR) & Dynamic Liquidity Splitting Engine.

Solves the multi-venue convex execution problem:
Minimizes total effective net cost (book price walk-through + non-linear slippage/impact + exchange taker fees)
across heterogeneous liquidity pools (e.g. Binance, Coinbase, Kraken, OKX) with maker rebate harvesting,
latency jitter compensation, and parallel atomic child slice dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SOROptimizationObjective(str, Enum):
    """Optimization objective for multi-venue order splitting."""
    MIN_TOTAL_NET_COST = "min_total_net_cost"      # Minimize purchase price + fees + slippage
    MAX_MAKER_HARVESTING = "max_maker_harvesting"  # Maximize maker rebate capture with passive depth
    MIN_EXECUTION_RISK = "min_execution_risk"      # Minimize duration & market impact risk (urgency)
    BALANCED_CONVEX = "balanced_convex"            # Weighted balance between price, fee, and dispersion


class VenueLiquidityType(str, Enum):
    MAKER = "MAKER"
    TAKER = "TAKER"


@dataclass
class VenueBookDepth:
    """Multi-level depth profile for a specific venue."""
    venue: str
    symbol: str
    bids: List[Tuple[float, float]]  # [(price, qty), ...] sorted high to low
    asks: List[Tuple[float, float]]  # [(price, qty), ...] sorted low to high
    maker_fee_bps: float = -1.0      # -1 bps maker rebate
    taker_fee_bps: float = 6.0       # 6 bps taker fee
    latency_ms: float = 5.0          # Estimated roundtrip network latency
    fill_rate_history: float = 0.98  # Historical passive fill probability [0, 1]

    def get_total_ask_liquidity(self, max_levels: int = 10) -> float:
        return sum(q for _, q in self.asks[:max_levels])

    def get_total_bid_liquidity(self, max_levels: int = 10) -> float:
        return sum(q for _, q in self.bids[:max_levels])


@dataclass(frozen=True)
class SyntheticChildAllocation:
    """Optimized child slice allocated to a target venue."""
    venue: str
    symbol: str
    side: str
    allocated_quantity: float
    allocation_percentage: float
    effective_unit_price: float
    expected_slippage_bps: float
    expected_fee_bps: float
    net_effective_cost: float
    liquidity_type: VenueLiquidityType
    dispatch_priority: int  # 1 = immediate, 2 = secondary, etc.
    estimated_arrival_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue,
            "symbol": self.symbol,
            "side": self.side,
            "allocated_quantity": float(self.allocated_quantity),
            "allocation_percentage": float(self.allocation_percentage),
            "effective_unit_price": float(self.effective_unit_price),
            "expected_slippage_bps": float(self.expected_slippage_bps),
            "expected_fee_bps": float(self.expected_fee_bps),
            "net_effective_cost": float(self.net_effective_cost),
            "liquidity_type": self.liquidity_type.value,
            "dispatch_priority": int(self.dispatch_priority),
            "estimated_arrival_latency_ms": float(self.estimated_arrival_latency_ms),
        }


@dataclass(frozen=True)
class DynamicLiquidityRoutingPlan:
    """Full synthetic multi-venue smart routing plan."""
    plan_id: str
    symbol: str
    side: str
    parent_quantity: float
    objective: SOROptimizationObjective
    timestamp_utc: datetime
    child_allocations: List[SyntheticChildAllocation]
    blended_vwap: float
    blended_net_price: float
    total_expected_fees: float
    expected_savings_bps: float  # Savings vs naive single-venue best-price routing
    is_fully_allocated: bool
    unallocated_quantity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "symbol": self.symbol,
            "side": self.side,
            "parent_quantity": float(self.parent_quantity),
            "objective": self.objective.value,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "child_allocations": [a.to_dict() for a in self.child_allocations],
            "blended_vwap": float(self.blended_vwap),
            "blended_net_price": float(self.blended_net_price),
            "total_expected_fees": float(self.total_expected_fees),
            "expected_savings_bps": float(self.expected_savings_bps),
            "is_fully_allocated": bool(self.is_fully_allocated),
            "unallocated_quantity": float(self.unallocated_quantity),
        }


class SyntheticSmartOrderRouter:
    """Institutional Cross-Venue Synthetic Smart Order Router & Convex Splitter.

    Features:
    1. Aggregates multi-venue L2 order book depth profiles with real-time fee schedules and latencies.
    2. Convex Multi-Venue Splitting Algorithm:
       Walks multi-venue marginal cost curves simultaneously across venues.
       For BUY parent order:
         Cost(q_v) = Integral_{0}^{q_v} P_v(x) dx + Fee_v(q_v)
         MarginalCost_v(q_v) = P_v(q_v) * (1 + fee_rate_v) + Impact_v(q_v)
       At each iteration, water-filling / greedy marginal allocation allocates increment dq
       to the venue with the lowest current marginal net cost.
    3. Latency Jitter Compensation: Staggers dispatch timestamps so child orders arrive simultaneously
       at respective exchange matching engines (Lead-Lag synchronization).
    4. Maker Rebate Harvesting: Identifies opportunities to post passive maker depth ahead of market arrival.
    5. Savings Benchmark: Calculates basis points saved compared to naive single-venue routing.
    """

    def __init__(
        self,
        default_objective: SOROptimizationObjective = SOROptimizationObjective.MIN_TOTAL_NET_COST,
        step_granularity: float = 0.05,  # Incremental allocation step (in base currency or percent)
        impact_coefficient: float = 0.15,
    ) -> None:
        self.default_objective = default_objective
        self.step_granularity = float(step_granularity)
        self.impact_coefficient = float(impact_coefficient)
        self._venue_depths: Dict[str, VenueBookDepth] = {}

    def register_venue_depth(self, venue_depth: VenueBookDepth) -> None:
        """Register or update venue depth profile."""
        self._venue_depths[venue_depth.venue.lower()] = venue_depth

    def clear_venues(self) -> None:
        """Clear all registered venues."""
        self._venue_depths.clear()

    def plan_liquidity_split(
        self,
        symbol: str,
        side: str,
        parent_quantity: float,
        objective: Optional[SOROptimizationObjective] = None,
        max_venues: int = 5,
    ) -> DynamicLiquidityRoutingPlan:
        """Compute optimal multi-venue child allocation.

        Args:
            symbol: Ticker symbol (e.g. BTC-USD)
            side: BUY or SELL
            parent_quantity: Total parent order volume
            objective: Optimization goal
            max_venues: Max distinct venues allowed in allocation

        Returns:
            DynamicLiquidityRoutingPlan with comprehensive metrics.
        """
        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side {side}, must be BUY or SELL")
        if parent_quantity <= 0:
            raise ValueError(f"Parent quantity must be positive, got {parent_quantity}")

        active_venues = [
            v for v in self._venue_depths.values()
            if v.symbol.upper() == symbol.upper()
        ]
        if not active_venues:
            raise ValueError(f"No liquidity venues available for symbol {symbol}")

        obj = objective or self.default_objective

        # 1. Execute Marginal Convex Optimization
        allocations_map = self._optimize_marginal_allocation(
            venues=active_venues,
            side=side_norm,
            total_qty=parent_quantity,
            objective=obj,
        )

        # 2. Build detailed child allocations with latency synchronization
        child_allocations: List[SyntheticChildAllocation] = []
        allocated_total = sum(allocations_map.values())
        max_latency = max(v.latency_ms for v in active_venues) if active_venues else 0.0

        # Sort venues by allocated quantity descending
        sorted_allocs = sorted(allocations_map.items(), key=lambda kv: kv[1], reverse=True)
        # Filter top max_venues
        if len(sorted_allocs) > max_venues:
            top_venues = dict(sorted_allocs[:max_venues])
            # Re-normalize or reallocate remainder
            s = sum(top_venues.values())
            if s > 0:
                scale = parent_quantity / s
                sorted_allocs = [(v, q * scale) for v, q in sorted_allocs[:max_venues]]

        total_gross_notional = 0.0
        total_net_notional = 0.0
        total_fees = 0.0

        for priority, (v_name, q) in enumerate(sorted_allocs, start=1):
            if q <= 1e-6:
                continue
            v_depth = self._venue_depths[v_name]
            vwap, slippage_bps, impact_bps = self._simulate_venue_fill(
                v_depth, side_norm, q
            )

            is_maker = (obj == SOROptimizationObjective.MAX_MAKER_HARVESTING)
            fee_bps = v_depth.maker_fee_bps if is_maker else v_depth.taker_fee_bps
            fee_rate = fee_bps / 10000.0

            gross_cost = vwap * q
            fee_amount = gross_cost * fee_rate

            if side_norm == "BUY":
                net_cost = gross_cost + fee_amount
                net_unit_price = net_cost / q
            else:
                net_cost = gross_cost - fee_amount
                net_unit_price = net_cost / q

            total_gross_notional += gross_cost
            total_net_notional += net_cost
            total_fees += fee_amount

            # Dispatch priority based on latency (longest latency dispatched first)
            lead_lag_priority = int(max(1, int(round((max_latency - v_depth.latency_ms) / 5.0)) + 1))

            child_allocations.append(
                SyntheticChildAllocation(
                    venue=v_depth.venue,
                    symbol=symbol,
                    side=side_norm,
                    allocated_quantity=round(q, 6),
                    allocation_percentage=round((q / parent_quantity) * 100.0, 2),
                    effective_unit_price=round(vwap, 4),
                    expected_slippage_bps=round(slippage_bps, 2),
                    expected_fee_bps=round(fee_bps, 2),
                    net_effective_cost=round(net_cost, 4),
                    liquidity_type=VenueLiquidityType.MAKER if is_maker else VenueLiquidityType.TAKER,
                    dispatch_priority=lead_lag_priority,
                    estimated_arrival_latency_ms=v_depth.latency_ms,
                )
            )

        # 3. Benchmark against Naive Single-Venue Routing (Best raw quote venue taking 100%)
        naive_net_cost = self._evaluate_naive_single_venue_cost(
            active_venues, side_norm, parent_quantity, obj
        )
        savings_bps = 0.0
        if naive_net_cost > 0:
            if side_norm == "BUY":
                savings_bps = ((naive_net_cost - total_net_notional) / naive_net_cost) * 10000.0
            else:
                savings_bps = ((total_net_notional - naive_net_cost) / naive_net_cost) * 10000.0

        blended_vwap = (total_gross_notional / allocated_total) if allocated_total > 0 else 0.0
        blended_net_price = (total_net_notional / allocated_total) if allocated_total > 0 else 0.0

        unallocated = max(0.0, parent_quantity - allocated_total)

        import uuid
        return DynamicLiquidityRoutingPlan(
            plan_id=f"sor_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side_norm,
            parent_quantity=parent_quantity,
            objective=obj,
            timestamp_utc=datetime.now(timezone.utc),
            child_allocations=child_allocations,
            blended_vwap=round(blended_vwap, 4),
            blended_net_price=round(blended_net_price, 4),
            total_expected_fees=round(total_fees, 4),
            expected_savings_bps=round(max(-100.0, savings_bps), 2),
            is_fully_allocated=(unallocated < 1e-5),
            unallocated_quantity=round(unallocated, 6),
        )

    # -------------------------------------------------------------------------
    # Convex Optimization & Simulation Routines
    # -------------------------------------------------------------------------

    def _optimize_marginal_allocation(
        self,
        venues: List[VenueBookDepth],
        side: str,
        total_qty: float,
        objective: SOROptimizationObjective,
    ) -> Dict[str, float]:
        """Greedy discrete convex water-filling across venues' marginal cost curves."""
        step = max(total_qty / 100.0, min(self.step_granularity, total_qty))
        allocations = {v.venue.lower(): 0.0 for v in venues}
        remaining = total_qty

        while remaining > 1e-7:
            curr_step = min(step, remaining)
            best_venue: Optional[VenueBookDepth] = None
            best_marginal_score: float = float("inf") if side == "BUY" else float("-inf")

            for v in venues:
                v_name = v.venue.lower()
                current_v_qty = allocations[v_name]

                # Evaluate marginal cost of adding curr_step to this venue
                marginal_cost = self._evaluate_marginal_cost(
                    v, side, current_v_qty, curr_step, objective
                )

                if side == "BUY":
                    if marginal_cost < best_marginal_score:
                        best_marginal_score = marginal_cost
                        best_venue = v
                else:
                    if marginal_cost > best_marginal_score:
                        best_marginal_score = marginal_cost
                        best_venue = v

            if best_venue is None:
                # Distribute evenly if all books exhausted
                for v in venues:
                    allocations[v.venue.lower()] += remaining / len(venues)
                break

            allocations[best_venue.venue.lower()] += curr_step
            remaining -= curr_step

        return allocations

    def _evaluate_marginal_cost(
        self,
        venue: VenueBookDepth,
        side: str,
        existing_qty: float,
        incremental_qty: float,
        objective: SOROptimizationObjective,
    ) -> float:
        """Calculate marginal net cost of executing an incremental slice at the target venue."""
        # Cost of existing + inc
        new_total = existing_qty + incremental_qty
        vwap_new, slip_new, _ = self._simulate_venue_fill(venue, side, new_total)
        vwap_old, slip_old, _ = self._simulate_venue_fill(venue, side, existing_qty) if existing_qty > 0 else (vwap_new, 0.0, 0.0)

        is_maker = (objective == SOROptimizationObjective.MAX_MAKER_HARVESTING)
        fee_rate = (venue.maker_fee_bps if is_maker else venue.taker_fee_bps) / 10000.0

        # Marginal gross cost = (vwap_new * new_total) - (vwap_old * existing_qty)
        marginal_gross = (vwap_new * new_total) - (vwap_old * existing_qty)
        marginal_unit_price = marginal_gross / incremental_qty if incremental_qty > 0 else vwap_new

        if side == "BUY":
            marginal_net = marginal_unit_price * (1.0 + fee_rate)
            # Add dispersion penalty if BALANCED_CONVEX
            if objective == SOROptimizationObjective.BALANCED_CONVEX:
                marginal_net += 0.01 * slip_new
        else:
            marginal_net = marginal_unit_price * (1.0 - fee_rate)
            if objective == SOROptimizationObjective.BALANCED_CONVEX:
                marginal_net -= 0.01 * slip_new

        return marginal_net

    def _simulate_venue_fill(
        self,
        venue: VenueBookDepth,
        side: str,
        quantity: float,
    ) -> Tuple[float, float, float]:
        """Simulate book traversal fill against venue order book levels.

        Returns: (vwap, slippage_bps, impact_bps)
        """
        levels = venue.asks if side == "BUY" else venue.bids
        if not levels or quantity <= 0:
            top_p = levels[0][0] if levels else 0.0
            return (top_p, 0.0, 0.0)

        rem_q = quantity
        total_notional = 0.0
        filled_q = 0.0
        top_price = levels[0][0]

        for p, q in levels:
            if rem_q <= 0:
                break
            fill = min(rem_q, q)
            total_notional += p * fill
            filled_q += fill
            rem_q -= fill

        if rem_q > 0:
            # Swept through available levels: extrapolate last price with impact
            last_p = levels[-1][0]
            extra_impact = self.impact_coefficient * (rem_q / max(filled_q, 1.0)) * 0.01
            extrapolated_p = last_p * (1.0 + extra_impact) if side == "BUY" else last_p * (1.0 - extra_impact)
            total_notional += extrapolated_p * rem_q
            filled_q += rem_q

        vwap = total_notional / filled_q if filled_q > 0 else top_price

        # Slippage vs top of book
        if top_price > 0:
            if side == "BUY":
                slippage_bps = ((vwap - top_price) / top_price) * 10000.0
            else:
                slippage_bps = ((top_price - vwap) / top_price) * 10000.0
        else:
            slippage_bps = 0.0

        # Permanent/Temporary impact approximation
        impact_bps = abs(slippage_bps) * 0.5

        return (vwap, max(0.0, slippage_bps), impact_bps)

    def _evaluate_naive_single_venue_cost(
        self,
        venues: List[VenueBookDepth],
        side: str,
        quantity: float,
        objective: SOROptimizationObjective,
    ) -> float:
        """Evaluate total net execution cost if 100% of order were sent to best top-of-book venue."""
        # Find venue with best top of book quote
        best_v = None
        best_p = float("inf") if side == "BUY" else float("-inf")

        for v in venues:
            levels = v.asks if side == "BUY" else v.bids
            if levels:
                p = levels[0][0]
                if (side == "BUY" and p < best_p) or (side == "SELL" and p > best_p):
                    best_p = p
                    best_v = v

        if not best_v:
            return 0.0

        vwap, _, _ = self._simulate_venue_fill(best_v, side, quantity)
        is_maker = (objective == SOROptimizationObjective.MAX_MAKER_HARVESTING)
        fee_rate = (best_v.maker_fee_bps if is_maker else best_v.taker_fee_bps) / 10000.0

        gross = vwap * quantity
        return gross * (1.0 + fee_rate) if side == "BUY" else gross * (1.0 - fee_rate)
