"""Trading Intelligence Platform - Dynamic Order Book Liquidity Heatmap & High-Frequency OFI Engine.

Implements the Cont-Kukanov-Stoikov Order Flow Imbalance (OFI) framework, multi-level depth OFI,
spatial price-time liquidity heatmap binning, Gini liquidity concentration indexing, and passive
queue position fill probability estimation.
"""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("orderbook_heatmap")


class BBOState(BaseModel):
    """Snapshot of top-of-book best bid and offer."""

    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OFIMetric(BaseModel):
    """Order Flow Imbalance diagnostic record."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    instantaneous_ofi: float
    rolling_ofi_sum: float
    normalized_ofi_z: float
    bid_flow: float
    ask_flow: float
    predicted_tick_direction: str  # "UP", "DOWN", "NEUTRAL"


class DepthHeatmapGrid(BaseModel):
    """Spatial price-time volume intensity heatmap of limit order depth."""

    symbol: str
    price_levels: List[float]
    time_labels: List[str]
    bid_intensity_grid: List[List[float]]  # [time_idx][price_idx]
    ask_intensity_grid: List[List[float]]  # [time_idx][price_idx]
    liquidity_concentration_index: float   # Gini coefficient of liquidity spread (0=flat, 1=concentrated)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueuePositionReport(BaseModel):
    """Estimated queue ahead and fill progression for a passive limit order."""

    order_id: str
    symbol: str
    price: float
    order_size: float
    initial_queue_ahead: float
    current_queue_ahead: float
    filled_quantity: float
    fill_percentage: float
    is_filled: bool
    estimated_fill_probability: float


class OrderFlowImbalanceCalculator:
    """Calculates instantaneous and rolling Cont-Kukanov-Stoikov OFI from BBO updates."""

    def __init__(self, rolling_window: int = 50, z_threshold: float = 1.5) -> None:
        self.rolling_window = rolling_window
        self.z_threshold = z_threshold
        self.prev_bbo: Optional[BBOState] = None
        self.ofi_history: List[float] = []

    def compute_step(self, curr_bbo: BBOState) -> OFIMetric:
        """Compute instantaneous and normalized OFI from previous to current BBO state."""
        if self.prev_bbo is None:
            self.prev_bbo = curr_bbo
            return OFIMetric(
                timestamp=curr_bbo.timestamp,
                instantaneous_ofi=0.0,
                rolling_ofi_sum=0.0,
                normalized_ofi_z=0.0,
                bid_flow=0.0,
                ask_flow=0.0,
                predicted_tick_direction="NEUTRAL",
            )

        prev = self.prev_bbo

        # 1. Bid order flow I_b (Cont-Kukanov-Stoikov 2014)
        if curr_bbo.bid_price > prev.bid_price:
            i_b = curr_bbo.bid_size
        elif curr_bbo.bid_price == prev.bid_price:
            i_b = curr_bbo.bid_size - prev.bid_size
        else:
            i_b = -prev.bid_size

        # 2. Ask order flow I_a
        if curr_bbo.ask_price < prev.ask_price:
            i_a = -curr_bbo.ask_size
        elif curr_bbo.ask_price == prev.ask_price:
            i_a = curr_bbo.ask_size - prev.ask_size
        else:
            i_a = prev.ask_size

        # 3. Instantaneous OFI
        inst_ofi = float(i_b - i_a)
        self.ofi_history.append(inst_ofi)
        if len(self.ofi_history) > self.rolling_window * 3:
            self.ofi_history.pop(0)

        # 4. Rolling statistics and Z-score normalization
        recent_window = self.ofi_history[-self.rolling_window:]
        rolling_sum = float(np.sum(recent_window))
        mean_ofi = float(np.mean(recent_window))
        std_ofi = float(np.std(recent_window))

        if std_ofi > 1e-6:
            z_score = float((inst_ofi - mean_ofi) / std_ofi)
        else:
            z_score = 0.0

        # 5. Directional prediction
        direction = "NEUTRAL"
        if z_score >= self.z_threshold:
            direction = "UP"
        elif z_score <= -self.z_threshold:
            direction = "DOWN"

        self.prev_bbo = curr_bbo

        return OFIMetric(
            timestamp=curr_bbo.timestamp,
            instantaneous_ofi=round(inst_ofi, 4),
            rolling_ofi_sum=round(rolling_sum, 4),
            normalized_ofi_z=round(z_score, 3),
            bid_flow=round(i_b, 4),
            ask_flow=round(i_a, 4),
            predicted_tick_direction=direction,
        )

    def compute_multi_level_ofi(
        self,
        prev_levels: List[Tuple[float, float]],
        curr_levels: List[Tuple[float, float]],
        is_bid: bool = True,
        depth: int = 5,
    ) -> float:
        """Calculate depth-weighted Multi-Level Order Flow Imbalance across top K book tiers."""
        total_flow = 0.0
        max_k = min(depth, len(prev_levels), len(curr_levels))

        for k in range(max_k):
            weight = 1.0 / (k + 1)
            p_prev, q_prev = prev_levels[k]
            p_curr, q_curr = curr_levels[k]

            if is_bid:
                if p_curr > p_prev:
                    flow = q_curr
                elif p_curr == p_prev:
                    flow = q_curr - q_prev
                else:
                    flow = -q_prev
            else:
                if p_curr < p_prev:
                    flow = -q_curr
                elif p_curr == p_prev:
                    flow = q_curr - q_prev
                else:
                    flow = q_prev

            total_flow += weight * flow

        return float(round(total_flow, 4))


class OrderBookHeatmapEngine:
    """Maintains time-series depth snapshots and renders price-time intensity matrices."""

    def __init__(self, symbol: str, price_tick_size: float = 1.0, max_snapshots: int = 50) -> None:
        self.symbol = symbol.upper()
        self.price_tick_size = price_tick_size
        self.max_snapshots = max_snapshots
        # List of {"timestamp": str, "bids": [(p, q)], "asks": [(p, q)]}
        self.snapshots: List[Dict[str, Any]] = []

    def record_snapshot(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append a new L2 depth snapshot."""
        ts = timestamp or datetime.now(timezone.utc)
        self.snapshots.append({
            "timestamp": ts.isoformat(),
            "bids": bids,
            "asks": asks,
        })
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient measuring spatial concentration of limit order volume."""
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return float(((2.0 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals))) - ((n + 1.0) / n))

    def generate_heatmap(self, num_price_levels: int = 20) -> DepthHeatmapGrid:
        """Render a normalized spatial price-time volume intensity heatmap."""
        if not self.snapshots:
            return DepthHeatmapGrid(
                symbol=self.symbol,
                price_levels=[],
                time_labels=[],
                bid_intensity_grid=[],
                ask_intensity_grid=[],
                liquidity_concentration_index=0.0,
            )

        # Collect global min and max prices to define price bins
        all_prices: List[float] = []
        for s in self.snapshots:
            for item in s["bids"]:
                all_prices.append(float(item[0]))
            for item in s["asks"]:
                all_prices.append(float(item[0]))

        if not all_prices:
            all_prices = [100.0]

        min_p = min(all_prices)
        max_p = max(all_prices)
        if min_p == max_p:
            min_p -= 10.0
            max_p += 10.0

        price_bins = np.linspace(min_p, max_p, num_price_levels).tolist()
        time_labels = [s["timestamp"] for s in self.snapshots]

        bid_grid: List[List[float]] = []
        ask_grid: List[List[float]] = []
        all_volumes: List[float] = []

        for s in self.snapshots:
            row_bid = [0.0] * num_price_levels
            row_ask = [0.0] * num_price_levels

            for item in s["bids"]:
                p, q = float(item[0]), float(item[1])
                # Map to nearest bin
                idx = int(np.argmin([abs(p - pb) for pb in price_bins]))
                row_bid[idx] += q
                all_volumes.append(q)

            for item in s["asks"]:
                p, q = float(item[0]), float(item[1])
                idx = int(np.argmin([abs(p - pb) for pb in price_bins]))
                row_ask[idx] += q
                all_volumes.append(q)

            bid_grid.append([round(v, 4) for v in row_bid])
            ask_grid.append([round(v, 4) for v in row_ask])

        gini = self._compute_gini_coefficient(np.array(all_volumes, dtype=np.float64))

        return DepthHeatmapGrid(
            symbol=self.symbol,
            price_levels=[round(p, 2) for p in price_bins],
            time_labels=time_labels,
            bid_intensity_grid=bid_grid,
            ask_intensity_grid=ask_grid,
            liquidity_concentration_index=round(gini, 3),
        )


class QueuePositionEstimator:
    """Estimates queue ahead, queue drain velocity, and fill probability for passive limit orders."""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        price: float,
        order_size: float,
        initial_depth_ahead: float,
    ) -> None:
        self.order_id = order_id
        self.symbol = symbol
        self.price = price
        self.order_size = order_size
        self.initial_queue = initial_depth_ahead
        self.current_queue = initial_depth_ahead
        self.filled_quantity = 0.0

    def process_trade(self, trade_price: float, trade_size: float) -> QueuePositionReport:
        """Process aggressive trade fill at the limit order price and advance queue."""
        if trade_price == self.price:
            if self.current_queue > 0:
                drain = min(self.current_queue, trade_size)
                self.current_queue -= drain
                leftover_trade = trade_size - drain
            else:
                leftover_trade = trade_size

            # If queue ahead is exhausted, trade matches against our order
            if self.current_queue <= 0 and leftover_trade > 0:
                fill_amt = min(self.order_size - self.filled_quantity, leftover_trade)
                self.filled_quantity += fill_amt

        fill_pct = (self.filled_quantity / max(self.order_size, 1e-6)) * 100.0
        is_filled = self.filled_quantity >= self.order_size

        # Fill probability estimation based on remaining queue
        if is_filled:
            prob = 1.0
        elif self.initial_queue > 0:
            prob = max(0.05, min(0.99, 1.0 - (self.current_queue / self.initial_queue)))
        else:
            prob = 0.95

        return QueuePositionReport(
            order_id=self.order_id,
            symbol=self.symbol,
            price=self.price,
            order_size=self.order_size,
            initial_queue_ahead=self.initial_queue,
            current_queue_ahead=round(self.current_queue, 4),
            filled_quantity=round(self.filled_quantity, 4),
            fill_percentage=round(fill_pct, 2),
            is_filled=is_filled,
            estimated_fill_probability=round(prob, 2),
        )
