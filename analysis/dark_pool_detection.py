"""Trading Intelligence Platform - Real-Time Dark Pool & Hidden Liquidity Detection Engine.

Detects hidden liquidity signatures, native and synthetic iceberg order replenishment,
midpoint dark pool crossing prints, and algorithmic block parent slices using order book
depth transitions and trade stream microstructural analytics.
"""

from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple
import uuid
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("dark_pool_detection")


class HiddenLiquidityType(str, Enum):
    """Classification of detected hidden or non-displayed liquidity."""

    ICEBERG_BID = "ICEBERG_BID"                           # Hidden bid replenishment absorbing sell orders
    ICEBERG_ASK = "ICEBERG_ASK"                           # Hidden ask replenishment absorbing buy orders
    DARK_CROSS_MIDPOINT = "DARK_CROSS_MIDPOINT"           # Trade executed inside spread at or near midpoint
    BLOCK_ALGO_SLICE = "BLOCK_ALGO_SLICE"                 # Recurring lot size indicative of algorithmic parent slicer
    OFF_BOOK_INTERNALIZATION = "OFF_BOOK_INTERNALIZATION" # Execution bypassing public visible order book queue


class IcebergSignature(BaseModel):
    """Detected active iceberg order with visible and estimated hidden parameters."""

    signature_id: str = Field(default_factory=lambda: f"ice_{uuid.uuid4().hex[:8]}")
    symbol: str
    venue: str
    side: str  # "BUY" or "SELL"
    price_level: float
    visible_display_qty: float
    cumulative_executed_qty: float
    replenishment_count: int
    estimated_hidden_qty: float
    estimated_total_parent_qty: float
    confidence_score: float  # 0.0 to 1.0
    first_detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_replenished_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    @property
    def total_executed_value_usd(self) -> float:
        return self.cumulative_executed_qty * self.price_level

    @property
    def estimated_hidden_value_usd(self) -> float:
        return self.estimated_hidden_qty * self.price_level


class DarkPrintRecord(BaseModel):
    """Telemetry for a single trade execution identified as dark, hidden, or internalized."""

    print_id: str = Field(default_factory=lambda: f"dark_{uuid.uuid4().hex[:8]}")
    symbol: str
    venue: str
    price: float
    quantity: float
    best_bid_at_time: float
    best_ask_at_time: float
    midpoint_at_time: float
    spread_bps_at_time: float
    liquidity_type: HiddenLiquidityType
    price_improvement_bps: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def notional_value_usd(self) -> float:
        return self.price * self.quantity


class DarkPoolDetectionReport(BaseModel):
    """Consolidated market microstructure report on hidden liquidity and dark volume share."""

    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active_icebergs: List[IcebergSignature] = []
    recent_dark_prints: List[DarkPrintRecord] = []
    aggregate_hidden_bid_depth_usd: float = 0.0
    aggregate_hidden_ask_depth_usd: float = 0.0
    total_lit_volume_usd: float = 0.0
    total_dark_volume_usd: float = 0.0
    dark_volume_share_pct: float = 0.0
    market_impact_implication: str = "BALANCED"


class LevelDepthTracker:
    """Tracks state and replenishment transitions at a single price level."""

    def __init__(self, price: float, side: str, initial_qty: float, timestamp: datetime) -> None:
        self.price = price
        self.side = side
        self.current_qty = initial_qty
        self.last_visible_qty = initial_qty
        self.last_update_time = timestamp
        self.replenishment_count = 0
        self.cumulative_replenished_qty = 0.0
        self.cumulative_executed_qty = 0.0
        self.replenishment_times: List[datetime] = []


class DarkPoolDetectionEngine:
    """Real-time microstructural detector for Iceberg orders, dark crossing, and hidden liquidity."""

    def __init__(
        self,
        replenishment_window_seconds: float = 2.0,
        min_replenishments: int = 2,
        min_hidden_ratio: float = 1.30,
        price_improvement_threshold_bps: float = 0.50,
        max_history_prints: int = 500,
    ) -> None:
        self.replenishment_window_seconds = replenishment_window_seconds
        self.min_replenishments = min_replenishments
        self.min_hidden_ratio = min_hidden_ratio
        self.price_improvement_threshold_bps = price_improvement_threshold_bps

        # Internal state
        # symbol -> side -> price -> LevelDepthTracker
        self._level_trackers: Dict[str, Dict[str, Dict[float, LevelDepthTracker]]] = {}
        # symbol -> active IcebergSignature
        self._active_icebergs: Dict[str, Dict[str, IcebergSignature]] = {}
        # symbol -> deque of DarkPrintRecord
        self._dark_prints: Dict[str, Deque[DarkPrintRecord]] = {}
        # symbol -> volume accumulators
        self._lit_volume: Dict[str, float] = {}
        self._dark_volume: Dict[str, float] = {}

    def _ensure_symbol_storage(self, symbol: str) -> None:
        if symbol not in self._level_trackers:
            self._level_trackers[symbol] = {"BUY": {}, "SELL": {}}
            self._active_icebergs[symbol] = {}
            self._dark_prints[symbol] = deque(maxlen=500)
            self._lit_volume[symbol] = 0.0
            self._dark_volume[symbol] = 0.0

    def process_book_depth_update(
        self,
        symbol: str,
        venue: str,
        side: str,  # "BUY" or "SELL"
        price: float,
        new_quantity: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[IcebergSignature]:
        """Process an incremental order book depth update at a given price level.

        Identifies replenishments: quantity increasing at an existing price level shortly
        after trades or depth consumption.
        """
        self._ensure_symbol_storage(symbol)
        ts = timestamp or datetime.now(timezone.utc)
        side_key = side.upper()
        trackers = self._level_trackers[symbol][side_key]

        if price not in trackers:
            trackers[price] = LevelDepthTracker(price=price, side=side_key, initial_qty=new_quantity, timestamp=ts)
            return None

        tracker = trackers[price]
        old_qty = tracker.current_qty
        delta_t = (ts - tracker.last_update_time).total_seconds()
        tracker.last_update_time = ts
        tracker.current_qty = new_quantity

        # Check for replenishment: depth suddenly increased at the same price
        if new_quantity > old_qty and delta_t <= self.replenishment_window_seconds:
            replenished_qty = new_quantity - old_qty
            tracker.replenishment_count += 1
            tracker.cumulative_replenished_qty += replenished_qty
            tracker.replenishment_times.append(ts)

            # Check if this qualifies as an iceberg signature
            if tracker.replenishment_count >= self.min_replenishments:
                # Estimate hidden parameters
                avg_display = tracker.last_visible_qty
                cum_exec = tracker.cumulative_executed_qty + tracker.cumulative_replenished_qty
                # Projected remaining tranches using geometric decay expectation
                est_remaining_tranches = max(1, tracker.replenishment_count)
                est_hidden_qty = est_remaining_tranches * avg_display
                est_total = cum_exec + est_hidden_qty

                # Confidence calculation based on count and consistency
                conf = min(0.98, 0.50 + 0.12 * tracker.replenishment_count)

                sig_side = "BUY" if side_key == "BUY" else "SELL"
                ice_id = f"{symbol}_{sig_side}_{price}"

                signature = IcebergSignature(
                    signature_id=ice_id,
                    symbol=symbol,
                    venue=venue,
                    side=sig_side,
                    price_level=price,
                    visible_display_qty=new_quantity,
                    cumulative_executed_qty=round(cum_exec, 4),
                    replenishment_count=tracker.replenishment_count,
                    estimated_hidden_qty=round(est_hidden_qty, 4),
                    estimated_total_parent_qty=round(est_total, 4),
                    confidence_score=round(conf, 3),
                    first_detected_at=tracker.replenishment_times[0],
                    last_replenished_at=ts,
                    is_active=True,
                )
                self._active_icebergs[symbol][ice_id] = signature
                logger.info(
                    f"Iceberg detected on {symbol} {venue} at ${price:.2f} "
                    f"[{sig_side}], replenishments={tracker.replenishment_count}, conf={conf:.2f}"
                )
                return signature

        return None

    def process_trade_print(
        self,
        symbol: str,
        venue: str,
        price: float,
        quantity: float,
        best_bid: float,
        best_ask: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[DarkPrintRecord]:
        """Analyze an incoming trade execution print for dark pool, midpoint, or iceberg execution."""
        self._ensure_symbol_storage(symbol)
        ts = timestamp or datetime.now(timezone.utc)
        notional = price * quantity

        # Update level tracker execution tallies
        side_key = "BUY" if abs(price - best_ask) <= abs(price - best_bid) else "SELL"
        if price in self._level_trackers[symbol][side_key]:
            self._level_trackers[symbol][side_key][price].cumulative_executed_qty += quantity

        if best_bid <= 0 or best_ask <= best_bid:
            self._lit_volume[symbol] += notional
            return None

        midpoint = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid
        spread_bps = (spread / midpoint) * 10000.0

        # Check for trade inside prevailing NBBO spread (dark cross or internal midpoint match)
        if best_bid < price < best_ask:
            # Price improvement
            if price <= midpoint:
                improvement_bps = ((best_ask - price) / best_ask) * 10000.0
            else:
                improvement_bps = ((price - best_bid) / best_bid) * 10000.0

            if abs(price - midpoint) <= 0.05 * spread:
                liquidity_type = HiddenLiquidityType.DARK_CROSS_MIDPOINT
            else:
                liquidity_type = HiddenLiquidityType.OFF_BOOK_INTERNALIZATION


            record = DarkPrintRecord(
                symbol=symbol,
                venue=venue,
                price=price,
                quantity=quantity,
                best_bid_at_time=best_bid,
                best_ask_at_time=best_ask,
                midpoint_at_time=midpoint,
                spread_bps_at_time=round(spread_bps, 2),
                liquidity_type=liquidity_type,
                price_improvement_bps=round(improvement_bps, 2),
                timestamp=ts,
            )
            self._dark_prints[symbol].append(record)
            self._dark_volume[symbol] += notional
            logger.info(
                f"Dark cross detected: {symbol} {venue} {quantity:.4f} @ ${price:.2f} "
                f"(spread={spread_bps:.1f}bps, improvement={improvement_bps:.1f}bps)"
            )
            return record

        # Lit execution
        self._lit_volume[symbol] += notional
        return None

    def detect_icebergs(self, symbol: str) -> List[IcebergSignature]:
        """Return all active iceberg signatures for a given symbol."""
        self._ensure_symbol_storage(symbol)
        return [sig for sig in self._active_icebergs[symbol].values() if sig.is_active]

    def generate_report(self, symbol: str) -> DarkPoolDetectionReport:
        """Synthesize consolidated hidden liquidity and dark pool analytics report."""
        self._ensure_symbol_storage(symbol)
        active_ice = self.detect_icebergs(symbol)
        recent_prints = list(self._dark_prints[symbol])[-50:]  # Most recent 50

        hidden_bid_usd = sum(
            sig.estimated_hidden_value_usd for sig in active_ice if sig.side == "BUY"
        )
        hidden_ask_usd = sum(
            sig.estimated_hidden_value_usd for sig in active_ice if sig.side == "SELL"
        )

        lit_vol = self._lit_volume[symbol]
        dark_vol = self._dark_volume[symbol]
        total_vol = lit_vol + dark_vol
        dark_share = (dark_vol / total_vol * 100.0) if total_vol > 0 else 0.0

        # Market impact implication
        if hidden_bid_usd > 2.0 * max(1.0, hidden_ask_usd):
            implication = "HEAVY_BUYING_ABSORPTION"
        elif hidden_ask_usd > 2.0 * max(1.0, hidden_bid_usd):
            implication = "HEAVY_SELLING_ABSORPTION"
        elif dark_share > 30.0:
            implication = "HIGH_DARK_CROSS_ACTIVITY"
        else:
            implication = "BALANCED"

        return DarkPoolDetectionReport(
            symbol=symbol,
            active_icebergs=active_ice,
            recent_dark_prints=recent_prints,
            aggregate_hidden_bid_depth_usd=round(hidden_bid_usd, 2),
            aggregate_hidden_ask_depth_usd=round(hidden_ask_usd, 2),
            total_lit_volume_usd=round(lit_vol, 2),
            total_dark_volume_usd=round(dark_vol, 2),
            dark_volume_share_pct=round(dark_share, 2),
            market_impact_implication=implication,
        )

    def clear_stale_signatures(self, max_age_seconds: float = 60.0) -> int:
        """Deactivate iceberg signatures that have not seen replenishment activity."""
        now = datetime.now(timezone.utc)
        deactivated = 0
        for symbol, signatures in self._active_icebergs.items():
            for sig in signatures.values():
                if sig.is_active:
                    age = (now - sig.last_replenished_at).total_seconds()
                    if age > max_age_seconds:
                        sig.is_active = False
                        deactivated += 1
        return deactivated
