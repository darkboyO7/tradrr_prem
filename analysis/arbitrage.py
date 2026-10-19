"""Trading Intelligence Platform - Cross-Exchange Arbitrage Engine & Scanner.

Scans multi-venue consolidated order books in real-time, detecting spatial arbitrage
opportunities across crypto exchanges (Binance, Kraken, Coinbase), filtering for net
profitability after fees, tracking latency-decay risk, and logging venue performance statistics.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from core.logging import get_logger
from data.orderbook.consolidated import ArbitrageOpportunity, ConsolidatedOrderBook

logger = get_logger("arbitrage_scanner")


class ArbitrageStats(BaseModel):
    """Aggregate statistics for detected arbitrage opportunities."""

    total_detected: int = 0
    total_theoretical_pnl: float = 0.0
    highest_profit_bps: float = 0.0
    venue_pair_counts: Dict[str, int] = Field(default_factory=dict)
    active_opportunities_count: int = 0


class CrossExchangeArbitrageScanner:
    """Institutional scanner monitoring cross-venue spatial arbitrage."""

    def __init__(
        self,
        min_net_profit_bps: float = 3.0,
        min_notional_usd: float = 100.0,
        taker_fee_rates: Optional[Dict[str, float]] = None,
    ) -> None:
        self.min_net_profit_bps = min_net_profit_bps
        self.min_notional_usd = min_notional_usd
        self.taker_fee_rates = taker_fee_rates or {
            "binance": 0.00075,
            "kraken": 0.0016,
            "coinbase": 0.0020,
        }
        # symbol -> ConsolidatedOrderBook
        self._books: Dict[str, ConsolidatedOrderBook] = {}
        # Running statistics
        self._total_detected: int = 0
        self._total_theoretical_pnl: float = 0.0
        self._highest_profit_bps: float = 0.0
        self._venue_pair_counts: Dict[str, int] = defaultdict(int)
        self._active_opportunities: Dict[str, ArbitrageOpportunity] = {}

    def register_book(self, symbol: str, book: ConsolidatedOrderBook) -> None:
        """Register a consolidated order book for monitoring."""
        self._books[symbol.upper()] = book

    def get_book(self, symbol: str) -> Optional[ConsolidatedOrderBook]:
        """Retrieve consolidated book by symbol."""
        return self._books.get(symbol.upper())

    def scan_symbol(self, symbol: str) -> Optional[ArbitrageOpportunity]:
        """Scan a specific symbol's consolidated book for an actionable arbitrage opportunity."""
        book = self._books.get(symbol.upper())
        if not book:
            return None

        # Determine average taker fee rate across venues for detection
        avg_fee = sum(self.taker_fee_rates.values()) / max(len(self.taker_fee_rates), 1)
        opp = book.detect_arbitrage(
            taker_fee_pct=avg_fee,
            min_net_profit_bps=self.min_net_profit_bps,
        )

        if not opp:
            self._active_opportunities.pop(symbol.upper(), None)
            return None

        # Filter by minimum notional USD
        notional = opp.max_executable_volume * opp.buy_price
        if notional < self.min_notional_usd:
            self._active_opportunities.pop(symbol.upper(), None)
            return None

        # Record metrics
        pair_key = f"{opp.buy_venue}->{opp.sell_venue}"
        self._total_detected += 1
        self._total_theoretical_pnl += opp.estimated_net_pnl
        if opp.net_profit_bps > self._highest_profit_bps:
            self._highest_profit_bps = opp.net_profit_bps
        self._venue_pair_counts[pair_key] += 1
        self._active_opportunities[symbol.upper()] = opp

        logger.info(
            "Arbitrage opportunity detected",
            symbol=symbol,
            pair=pair_key,
            net_bps=opp.net_profit_bps,
            est_pnl=opp.estimated_net_pnl,
            notional=round(notional, 2),
        )

        return opp

    def scan_all(self) -> List[ArbitrageOpportunity]:
        """Scan all registered symbols for arbitrage opportunities."""
        actionable: List[ArbitrageOpportunity] = []
        for symbol in self._books.keys():
            opp = self.scan_symbol(symbol)
            if opp:
                actionable.append(opp)
        return actionable

    def get_stats(self) -> ArbitrageStats:
        """Return running telemetry and metrics on scanned arbitrage."""
        return ArbitrageStats(
            total_detected=self._total_detected,
            total_theoretical_pnl=round(self._total_theoretical_pnl, 2),
            highest_profit_bps=round(self._highest_profit_bps, 2),
            venue_pair_counts=dict(self._venue_pair_counts),
            active_opportunities_count=len(self._active_opportunities),
        )
