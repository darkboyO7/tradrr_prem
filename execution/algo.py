"""Trading Intelligence Platform - Algorithmic Execution Engine (TWAP / VWAP).

Implements institutional slicing algorithms:
- Time-Weighted Average Price (TWAP) with Gaussian/uniform time jitter and size randomization.
- Volume-Weighted Average Price (VWAP) weighted by intra-day volume curves.
- Execution Quality Benchmarking (Arrival Price Implementation Shortfall & Market VWAP tracking error).
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger
from execution.base import BaseExecutionClient
from execution.volume_profile import IntradayVolumeProfile, VolumeProfiler

logger = get_logger("algo_engine")


class AlgoOrderType(str, Enum):
    """Institutional execution algorithm types."""

    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price
    POV = "POV"    # Percentage of Volume participation


class AlgoSliceSchedule(BaseModel):
    """Individual planned execution slice within an algorithmic order."""

    slice_index: int
    time_offset_seconds: float
    quantity: float
    limit_price: Optional[float] = None
    executed: bool = False
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    execution_fee: Optional[float] = None


class AlgoExecutionPlan(BaseModel):
    """Execution schedule for a sliced parent order."""

    algo_id: str = Field(default_factory=lambda: f"alg_{uuid.uuid4().hex[:8]}")
    algo_type: AlgoOrderType
    symbol: str
    side: str  # BUY or SELL
    parent_quantity: float
    duration_seconds: float
    arrival_price: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    slices: List[AlgoSliceSchedule] = []


class AlgoExecutionReport(BaseModel):
    """Post-trade execution quality audit and implementation shortfall benchmark."""

    algo_id: str
    symbol: str
    side: str
    status: str
    parent_quantity: float
    filled_quantity: float
    arrival_price: float
    realized_vwap: float
    market_vwap: float
    arrival_slippage_bps: float = Field(
        description="Execution drag vs Arrival Price (positive value indicates adverse slippage)"
    )
    market_vwap_slippage_bps: float = Field(
        description="Tracking error vs Market VWAP benchmark"
    )
    total_fees_paid: float
    slices_executed_count: int
    slices_total_count: int


class AlgoEngine:
    """Institutional TWAP / VWAP execution schedule generator and dispatcher."""

    def __init__(self, volume_profiler: Optional[VolumeProfiler] = None) -> None:
        self.profiler = volume_profiler or VolumeProfiler()

    def create_twap_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        duration_minutes: float,
        num_slices: int = 10,
        arrival_price: float = 65000.0,
        time_jitter_pct: float = 0.15,
        size_jitter_pct: float = 0.10,
        random_seed: int = 42,
    ) -> AlgoExecutionPlan:
        """Create a randomized TWAP schedule to prevent predatory front-running."""
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if duration_minutes <= 0:
            raise ValueError(f"Duration must be positive, got {duration_minutes}")
        if num_slices <= 0:
            raise ValueError(f"num_slices must be > 0, got {num_slices}")

        side = side.upper()
        rng = np.random.default_rng(random_seed)
        total_duration_sec = duration_minutes * 60.0
        interval_sec = total_duration_sec / num_slices

        # Base slice size
        base_slice_qty = quantity / num_slices
        # Size randomization
        size_noises = rng.uniform(-size_jitter_pct, size_jitter_pct, size=num_slices)
        raw_slice_sizes = base_slice_qty * (1.0 + size_noises)
        # Normalize sum of slice sizes to match exact parent quantity
        slice_sizes = raw_slice_sizes * (quantity / raw_slice_sizes.sum())

        slices: List[AlgoSliceSchedule] = []
        for i in range(num_slices):
            base_time = i * interval_sec
            # Time jitter (except first slice which starts immediately at t=0)
            if i == 0:
                time_offset = 0.0
            else:
                jitter = rng.uniform(-time_jitter_pct * interval_sec, time_jitter_pct * interval_sec)
                time_offset = max(base_time + jitter, (i - 0.5) * interval_sec)

            slices.append(
                AlgoSliceSchedule(
                    slice_index=i,
                    time_offset_seconds=round(float(time_offset), 2),
                    quantity=round(float(slice_sizes[i]), 6),
                )
            )

        # Fix minor rounding residual on final slice
        diff = round(quantity - sum(s.quantity for s in slices), 6)
        if diff != 0:
            slices[-1].quantity = round(slices[-1].quantity + diff, 6)

        return AlgoExecutionPlan(
            algo_type=AlgoOrderType.TWAP,
            symbol=symbol.upper(),
            side=side,
            parent_quantity=quantity,
            duration_seconds=total_duration_sec,
            arrival_price=arrival_price,
            slices=slices,
        )

    def create_vwap_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        duration_minutes: float,
        num_slices: int = 12,
        arrival_price: float = 65000.0,
        volume_profile: Optional[IntradayVolumeProfile] = None,
        size_jitter_pct: float = 0.05,
        random_seed: int = 42,
    ) -> AlgoExecutionPlan:
        """Create a volume-weighted VWAP execution schedule based on expected intraday volume."""
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        if duration_minutes <= 0:
            raise ValueError(f"Duration must be positive, got {duration_minutes}")
        if num_slices <= 0:
            raise ValueError(f"num_slices must be > 0, got {num_slices}")

        side = side.upper()
        rng = np.random.default_rng(random_seed)
        total_duration_sec = duration_minutes * 60.0
        interval_sec = total_duration_sec / num_slices

        profile = volume_profile or self.profiler.get_canonical_profile(symbol)
        now = datetime.now(timezone.utc)
        duration_hours = max(duration_minutes / 60.0, 0.1)

        weights = self.profiler.get_slice_fractions_for_window(
            profile=profile,
            start_time=now,
            duration_hours=duration_hours,
            num_slices=num_slices,
        )

        raw_quantities = np.array(weights) * quantity
        if size_jitter_pct > 0:
            noises = rng.uniform(-size_jitter_pct, size_jitter_pct, size=num_slices)
            raw_quantities = raw_quantities * (1.0 + noises)
            raw_quantities = raw_quantities * (quantity / raw_quantities.sum())

        slices: List[AlgoSliceSchedule] = []
        for i in range(num_slices):
            time_offset = i * interval_sec
            slices.append(
                AlgoSliceSchedule(
                    slice_index=i,
                    time_offset_seconds=round(float(time_offset), 2),
                    quantity=round(float(raw_quantities[i]), 6),
                )
            )

        # Fix minor rounding on final slice
        diff = round(quantity - sum(s.quantity for s in slices), 6)
        if diff != 0:
            slices[-1].quantity = round(slices[-1].quantity + diff, 6)

        return AlgoExecutionPlan(
            algo_type=AlgoOrderType.VWAP,
            symbol=symbol.upper(),
            side=side,
            parent_quantity=quantity,
            duration_seconds=total_duration_sec,
            arrival_price=arrival_price,
            slices=slices,
        )

    async def execute_algo_plan(
        self,
        plan: AlgoExecutionPlan,
        client: BaseExecutionClient,
        simulated_prices: Optional[List[float]] = None,
        fast_simulation: bool = True,
    ) -> AlgoExecutionReport:
        """Dispatch child order slices and audit execution quality benchmarks.
        
        Args:
            plan: The planned AlgoExecutionPlan.
            client: Execution client (Simulated, REST, etc.).
            simulated_prices: Sequence of market fill prices for testing/simulation.
            fast_simulation: If True, skips actual asyncio.sleep delays between slices.
        """
        filled_qty = 0.0
        total_dollars = 0.0
        total_fees = 0.0
        executed_slices = 0

        # Use simulated price series or fallback to arrival price with minor brownian drift
        prices = simulated_prices or [plan.arrival_price] * len(plan.slices)
        market_vwap_dollars = 0.0
        market_vwap_vol = 0.0

        for idx, slc in enumerate(plan.slices):
            fill_price = prices[idx % len(prices)]
            market_vwap_dollars += fill_price * slc.quantity
            market_vwap_vol += slc.quantity

            if not fast_simulation and slc.time_offset_seconds > 0:
                await asyncio.sleep(0.01)

            res = await client.place_order(
                symbol=plan.symbol,
                side=plan.side,
                order_type="MARKET",
                quantity=slc.quantity,
                price=fill_price,
                client_order_id=f"{plan.algo_id}_{slc.slice_index}",
            )

            executed_qty = float(res.get("executed_quantity", slc.quantity))
            realized_price = float(res.get("price", fill_price))
            fee = float(res.get("fee", realized_price * executed_qty * 0.001))

            slc.executed = True
            slc.executed_price = realized_price
            slc.executed_quantity = executed_qty
            slc.execution_fee = fee

            filled_qty += executed_qty
            total_dollars += executed_qty * realized_price
            total_fees += fee
            executed_slices += 1

        realized_vwap = round(total_dollars / filled_qty, 4) if filled_qty > 0 else plan.arrival_price
        market_vwap = round(market_vwap_dollars / market_vwap_vol, 4) if market_vwap_vol > 0 else plan.arrival_price

        # Arrival Price Slippage:
        # BUY: (Realized - Arrival) / Arrival * 10,000 (paying more is positive slippage drag)
        # SELL: (Arrival - Realized) / Arrival * 10,000 (selling lower is positive slippage drag)
        if plan.side == "BUY":
            arrival_slip_bps = ((realized_vwap - plan.arrival_price) / plan.arrival_price) * 10_000.0
        else:
            arrival_slip_bps = ((plan.arrival_price - realized_vwap) / plan.arrival_price) * 10_000.0

        # Market VWAP Slippage: (Realized - Market_VWAP) / Market_VWAP * 10,000
        if plan.side == "BUY":
            vwap_slip_bps = ((realized_vwap - market_vwap) / market_vwap) * 10_000.0
        else:
            vwap_slip_bps = ((market_vwap - realized_vwap) / market_vwap) * 10_000.0

        logger.info(
            "Algorithmic execution completed",
            algo_id=plan.algo_id,
            type=plan.algo_type.value,
            filled=round(filled_qty, 6),
            realized_vwap=realized_vwap,
            arrival_slippage_bps=round(arrival_slip_bps, 2),
        )

        return AlgoExecutionReport(
            algo_id=plan.algo_id,
            symbol=plan.symbol,
            side=plan.side,
            status="COMPLETED" if filled_qty >= plan.parent_quantity * 0.999 else "PARTIALLY_FILLED",
            parent_quantity=plan.parent_quantity,
            filled_quantity=round(filled_qty, 6),
            arrival_price=plan.arrival_price,
            realized_vwap=realized_vwap,
            market_vwap=market_vwap,
            arrival_slippage_bps=round(arrival_slip_bps, 2),
            market_vwap_slippage_bps=round(vwap_slip_bps, 2),
            total_fees_paid=round(total_fees, 4),
            slices_executed_count=executed_slices,
            slices_total_count=len(plan.slices),
        )
