"""Trading Intelligence Platform - Level 2 Order Book & Microstructure Engine.

Maintains live sorted bid/ask depth in memory, calculates top-of-book metrics,
micro-price, spread basis points, and multi-level Order Book Imbalance (OBI).
"""

from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class OrderBookMetrics(BaseModel):
    """Snapshot of quantitative microstructure metrics derived from the L2 order book."""

    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    best_bid: float
    best_ask: float
    mid_price: float
    micro_price: float
    spread: float
    spread_bps: float
    bid_depth_top5: float
    ask_depth_top5: float
    order_book_imbalance: float = Field(
        ge=-1.0, le=1.0, description="Normalized OBI: (BidQty - AskQty) / (BidQty + AskQty)"
    )


class OrderBookL2:
    """In-memory limit order book tracking depth across price levels."""

    def __init__(self, symbol: str, max_depth_levels: int = 50) -> None:
        self.symbol = symbol.upper()
        self.max_depth_levels = max_depth_levels
        # price -> quantity (bids sorted descending, asks sorted ascending)
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self.last_update_id: Optional[int] = None
        self.last_updated_at: datetime = datetime.now(timezone.utc)

    @property
    def best_bid(self) -> Optional[float]:
        return max(self._bids.keys()) if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return min(self._asks.keys()) if self._asks else None

    @property
    def spread(self) -> float:
        bb = self.best_bid
        ba = self.best_ask
        return round(ba - bb, 8) if (bb is not None and ba is not None) else 0.0

    @property
    def mid_price(self) -> float:
        bb = self.best_bid
        ba = self.best_ask
        return round((ba + bb) / 2.0, 8) if (bb is not None and ba is not None) else 0.0

    def apply_snapshot(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        update_id: Optional[int] = None,
    ) -> None:
        """Replace current book with a full depth snapshot."""
        self._bids.clear()
        self._asks.clear()

        for price, qty in bids:
            if qty > 0:
                self._bids[float(price)] = float(qty)

        for price, qty in asks:
            if qty > 0:
                self._asks[float(price)] = float(qty)

        self._truncate_depth()
        self.last_update_id = update_id
        self.last_updated_at = datetime.now(timezone.utc)

    def apply_delta(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        update_id: Optional[int] = None,
    ) -> None:
        """Apply incremental level updates (qty=0 removes the level)."""
        for price, qty in bids:
            p = float(price)
            q = float(qty)
            if q <= 0:
                self._bids.pop(p, None)
            else:
                self._bids[p] = q

        for price, qty in asks:
            p = float(price)
            q = float(qty)
            if q <= 0:
                self._asks.pop(p, None)
            else:
                self._asks[p] = q

        self._truncate_depth()
        self.last_update_id = update_id
        self.last_updated_at = datetime.now(timezone.utc)

    def _truncate_depth(self) -> None:
        """Keep only top N price levels to bound memory and processing latency."""
        if len(self._bids) > self.max_depth_levels:
            sorted_bids = sorted(self._bids.keys(), reverse=True)[: self.max_depth_levels]
            self._bids = {p: self._bids[p] for p in sorted_bids}

        if len(self._asks) > self.max_depth_levels:
            sorted_asks = sorted(self._asks.keys())[: self.max_depth_levels]
            self._asks = {p: self._asks[p] for p in sorted_asks}

    def get_top_levels(self, depth: int = 5) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Return the top sorted bids and asks up to depth levels."""
        sorted_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[:depth]
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])[:depth]
        return sorted_bids, sorted_asks

    def get_bids(self, depth: int = 5) -> List[Tuple[float, float]]:
        """Return sorted bids descending up to depth levels."""
        return sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[:depth]

    def get_asks(self, depth: int = 5) -> List[Tuple[float, float]]:
        """Return sorted asks ascending up to depth levels."""
        return sorted(self._asks.items(), key=lambda x: x[0])[:depth]

    def compute_metrics(self, depth_levels: int = 5) -> Optional[OrderBookMetrics]:
        """Compute microstructural features from the current book state."""
        bb = self.best_bid
        ba = self.best_ask

        if bb is None or ba is None or bb >= ba:
            return None

        top_bids, top_asks = self.get_top_levels(depth=depth_levels)
        bid_qty_sum = sum(q for _, q in top_bids)
        ask_qty_sum = sum(q for _, q in top_asks)

        # 1. Spread and Spread in Basis Points
        spread = ba - bb
        mid = (ba + bb) / 2.0
        spread_bps = (spread / mid) * 10000.0 if mid > 0 else 0.0

        # 2. Volume-Weighted Micro-Price (Volume Weighted Top of Book)
        q_bb = self._bids[bb]
        q_ba = self._asks[ba]
        denom = q_bb + q_ba
        micro_price = ((ba * q_bb) + (bb * q_ba)) / denom if denom > 0 else mid

        # 3. Order Book Imbalance (OBI) across top N levels
        total_depth = bid_qty_sum + ask_qty_sum
        obi = (bid_qty_sum - ask_qty_sum) / total_depth if total_depth > 0 else 0.0

        return OrderBookMetrics(
            symbol=self.symbol,
            timestamp=self.last_updated_at,
            best_bid=round(bb, 8),
            best_ask=round(ba, 8),
            mid_price=round(mid, 8),
            micro_price=round(micro_price, 8),
            spread=round(spread, 8),
            spread_bps=round(spread_bps, 2),
            bid_depth_top5=round(bid_qty_sum, 4),
            ask_depth_top5=round(ask_qty_sum, 4),
            order_book_imbalance=round(obi, 4),
        )
