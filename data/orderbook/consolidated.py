"""Trading Intelligence Platform - Consolidated Order Book & Multi-Venue Depth Engine.

Aggregates Level 2 limit order books from multiple venues (e.g. Binance, Kraken, Coinbase),
calculates the Synthetic National Best Bid and Offer (Consolidated BBO), and detects
cross-exchange arbitrage opportunities with fee adjustments.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.logging import get_logger
from data.orderbook.l2_book import OrderBookL2

logger = get_logger("consolidated_orderbook")


class VenueQuote(BaseModel):
    """Represents a top-of-book bid or ask quote from a specific venue."""

    venue: str
    price: float
    size: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SyntheticBBO(BaseModel):
    """Synthetic Best Bid and Offer across all aggregated venues."""

    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    best_bid: Optional[float] = None
    best_bid_venue: Optional[str] = None
    best_bid_size: Optional[float] = None
    best_ask: Optional[float] = None
    best_ask_venue: Optional[str] = None
    best_ask_size: Optional[float] = None
    spread: Optional[float] = None
    spread_bps: Optional[float] = None
    is_crossed: bool = Field(
        default=False,
        description="True if best_bid >= best_ask (indicates cross-venue arbitrage condition)",
    )


class ArbitrageOpportunity(BaseModel):
    """Details of an executable cross-exchange spatial arbitrage opportunity."""

    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    buy_venue: str
    buy_price: float
    buy_size: float
    sell_venue: str
    sell_price: float
    sell_size: float
    max_executable_volume: float
    gross_spread: float
    gross_spread_bps: float
    net_spread: float
    net_profit_bps: float
    estimated_net_pnl: float


class ConsolidatedOrderBook:
    """Multi-venue order book aggregator and synthetic BBO engine."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        # venue_name -> OrderBookL2 instance
        self._venue_books: Dict[str, OrderBookL2] = {}

    def register_venue(self, venue: str, max_depth_levels: int = 50) -> OrderBookL2:
        """Register a new venue for book aggregation."""
        venue = venue.lower()
        if venue not in self._venue_books:
            self._venue_books[venue] = OrderBookL2(symbol=self.symbol, max_depth_levels=max_depth_levels)
        return self._venue_books[venue]

    def update_venue_snapshot(
        self,
        venue: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        update_id: Optional[int] = None,
    ) -> None:
        """Update full L2 depth snapshot for a specific venue."""
        book = self.register_venue(venue)
        book.apply_snapshot(bids=bids, asks=asks, update_id=update_id)

    def update_venue_delta(
        self,
        venue: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        update_id: Optional[int] = None,
    ) -> None:
        """Apply incremental L2 updates for a specific venue."""
        book = self.register_venue(venue)
        book.apply_delta(bids=bids, asks=asks, update_id=update_id)

    def get_venue_book(self, venue: str) -> Optional[OrderBookL2]:
        """Retrieve L2 order book for a single venue."""
        return self._venue_books.get(venue.lower())

    @property
    def venues(self) -> List[str]:
        """List all registered venue identifiers."""
        return list(self._venue_books.keys())

    def get_consolidated_bbo(self) -> SyntheticBBO:
        """Calculate the Synthetic Best Bid and Offer across all venues."""
        highest_bid: Optional[float] = None
        best_bid_venue: Optional[str] = None
        best_bid_size: Optional[float] = None

        lowest_ask: Optional[float] = None
        best_ask_venue: Optional[str] = None
        best_ask_size: Optional[float] = None

        for venue, book in self._venue_books.items():
            bb = book.best_bid
            if bb is not None:
                bb_size = book.get_bids(1)[0][1] if book.get_bids(1) else 0.0
                if highest_bid is None or bb > highest_bid:
                    highest_bid = bb
                    best_bid_venue = venue
                    best_bid_size = bb_size

            ba = book.best_ask
            if ba is not None:
                ba_size = book.get_asks(1)[0][1] if book.get_asks(1) else 0.0
                if lowest_ask is None or ba < lowest_ask:
                    lowest_ask = ba
                    best_ask_venue = venue
                    best_ask_size = ba_size

        spread: Optional[float] = None
        spread_bps: Optional[float] = None
        is_crossed = False

        if highest_bid is not None and lowest_ask is not None:
            spread = round(lowest_ask - highest_bid, 8)
            mid = (lowest_ask + highest_bid) / 2.0
            spread_bps = round((spread / mid) * 10_000.0, 4) if mid > 0 else 0.0
            is_crossed = highest_bid >= lowest_ask

        return SyntheticBBO(
            symbol=self.symbol,
            best_bid=highest_bid,
            best_bid_venue=best_bid_venue,
            best_bid_size=best_bid_size,
            best_ask=lowest_ask,
            best_ask_venue=best_ask_venue,
            best_ask_size=best_ask_size,
            spread=spread,
            spread_bps=spread_bps,
            is_crossed=is_crossed,
        )

    def get_aggregated_depth(
        self, levels: int = 10
    ) -> Dict[str, List[Tuple[float, float, str]]]:
        """Aggregate and sort price levels across all venues.
        
        Returns:
            Dict containing:
                "bids": sorted descending [(price, aggregated_size, venue), ...]
                "asks": sorted ascending [(price, aggregated_size, venue), ...]
        """
        all_bids: List[Tuple[float, float, str]] = []
        all_asks: List[Tuple[float, float, str]] = []

        for venue, book in self._venue_books.items():
            for p, q in book.get_bids(levels):
                all_bids.append((p, q, venue))
            for p, q in book.get_asks(levels):
                all_asks.append((p, q, venue))

        # Sort bids descending, asks ascending
        sorted_bids = sorted(all_bids, key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(all_asks, key=lambda x: x[0])[:levels]

        return {"bids": sorted_bids, "asks": sorted_asks}

    def detect_arbitrage(
        self,
        taker_fee_pct: float = 0.001,  # 10 bps default taker fee
        min_net_profit_bps: float = 2.0,
    ) -> Optional[ArbitrageOpportunity]:
        """Scan across venue books for executable spatial arbitrage opportunities.
        
        Arbitrage occurs when:
        SellVenue Bid > BuyVenue Ask
        Net PnL = (Sell_Bid * (1 - taker_fee)) - (Buy_Ask * (1 + taker_fee)) > 0
        """
        bbo = self.get_consolidated_bbo()
        if not bbo.is_crossed or bbo.best_bid is None or bbo.best_ask is None:
            return None

        # Best bid is higher than or equal to best ask across different venues
        if bbo.best_bid_venue == bbo.best_ask_venue:
            # Internal locked/crossed book on same venue, ignore
            return None

        sell_venue = bbo.best_bid_venue
        sell_price = bbo.best_bid
        sell_size = bbo.best_bid_size or 0.0

        buy_venue = bbo.best_ask_venue
        buy_price = bbo.best_ask
        buy_size = bbo.best_ask_size or 0.0

        gross_spread = round(sell_price - buy_price, 8)
        mid_price = (sell_price + buy_price) / 2.0
        gross_spread_bps = round((gross_spread / mid_price) * 10_000.0, 4)

        # Net calculations with taker fees on both legs
        net_sell_proceeds_per_unit = sell_price * (1.0 - taker_fee_pct)
        net_buy_cost_per_unit = buy_price * (1.0 + taker_fee_pct)
        net_profit_per_unit = net_sell_proceeds_per_unit - net_buy_cost_per_unit

        net_profit_bps = round((net_profit_per_unit / buy_price) * 10_000.0, 4)

        if net_profit_bps < min_net_profit_bps:
            return None

        max_volume = min(sell_size, buy_size)
        estimated_net_pnl = round(net_profit_per_unit * max_volume, 2)

        return ArbitrageOpportunity(
            symbol=self.symbol,
            buy_venue=buy_venue,
            buy_price=buy_price,
            buy_size=buy_size,
            sell_venue=sell_venue,
            sell_price=sell_price,
            sell_size=sell_size,
            max_executable_volume=max_volume,
            gross_spread=gross_spread,
            gross_spread_bps=gross_spread_bps,
            net_spread=round(net_profit_per_unit, 8),
            net_profit_bps=net_profit_bps,
            estimated_net_pnl=estimated_net_pnl,
        )
