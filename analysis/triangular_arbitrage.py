"""Trading Intelligence Platform - Real-Time Cross-Exchange Triangular & Synthetic Latency Arbitrage Engine.

High-throughput cyclic graph traversal, negative log-price Bellman-Ford cycle exploration,
and multi-venue synthetic exchange rate routing across crypto currency triangles
(e.g., USD -> BTC -> ETH -> USD) factoring in taker fee drag and depth-based slippage.
"""

from collections import defaultdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("triangular_arbitrage")


class TriangularLeg(BaseModel):
    """Individual trade execution leg of a triangular arbitrage loop."""

    leg_index: int
    from_asset: str
    to_asset: str
    symbol: str
    venue: str
    action: str  # "BUY" or "SELL"
    rate: float
    quote_price: float
    max_input_qty: float
    fee_bps: float
    slippage_bps: float


class TriangularArbitrageOpportunity(BaseModel):
    """Validated cyclic triangular arbitrage opportunity across multi-currency pairs."""

    opportunity_id: str = Field(default_factory=lambda: f"tri_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    triangle_path: List[str]  # e.g. ["USD", "BTC", "ETH", "USD"]
    venues_involved: List[str]
    is_cross_venue: bool
    gross_product: float
    net_product: float
    gross_return_pct: float
    net_return_pct: float
    net_return_bps: float
    total_fees_bps: float
    total_slippage_bps: float
    max_executable_capital_usd: float
    expected_net_profit_usd: float
    execution_urgency: str  # "HIGH", "CRITICAL"
    legs: List[TriangularLeg]


class RateGraphEdge(BaseModel):
    """Directed edge in the multi-currency exchange rate graph."""

    from_asset: str
    to_asset: str
    symbol: str
    venue: str
    rate: float
    effective_rate: float
    weight: float  # -ln(effective_rate)
    action: str
    fee_bps: float


class RateGraphData(BaseModel):
    """Visualizable multi-currency directed rate graph representation."""

    nodes: List[str]
    edges: List[RateGraphEdge]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriangularScanReport(BaseModel):
    """Comprehensive summary of scanned triangular loops and detected opportunities."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_currency_pairs_active: int
    total_triangles_evaluated: int
    opportunities_count: int
    best_opportunity: Optional[TriangularArbitrageOpportunity] = None
    opportunities: List[TriangularArbitrageOpportunity] = []


class TriangularArbitrageEngine:
    """Microsecond graph-theoretic engine detecting cyclic triangular and synthetic arbitrage."""

    DEFAULT_TAKER_FEES_BPS = {
        "binance": 7.5,
        "kraken": 16.0,
        "coinbase": 20.0,
        "DEFAULT": 15.0,
    }

    def __init__(
        self,
        default_slippage_bps: float = 2.0,
        min_net_profit_bps: float = 5.0,
        max_capital_cap_usd: float = 100000.0,
        custom_fee_rates_bps: Optional[Dict[str, float]] = None,
    ) -> None:
        self.default_slippage_bps = default_slippage_bps
        self.min_net_profit_bps = min_net_profit_bps
        self.max_capital_cap_usd = max_capital_cap_usd
        self.fee_rates_bps = {**self.DEFAULT_TAKER_FEES_BPS, **(custom_fee_rates_bps or {})}

        # Quote store: (venue, symbol) -> (bid, ask, bid_qty, ask_qty, timestamp)
        self._quotes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Cached active currencies and pairs
        self._currencies: Set[str] = set()

    def get_fee_bps(self, venue: str) -> float:
        """Retrieve taker fee rate in basis points for a venue."""
        return self.fee_rates_bps.get(venue.lower(), self.fee_rates_bps["DEFAULT"])

    @staticmethod
    def parse_symbol(symbol: str) -> Tuple[str, str]:
        """Parse base and quote asset from canonical symbol (e.g. 'BTC-USD', 'ETH/BTC')."""
        clean = symbol.upper().replace("/", "-")
        if "-" in clean:
            parts = clean.split("-")
            return parts[0], parts[1]
        # Fallback for concatenated symbols
        for sep_len in [3, 4]:
            if len(clean) >= 6:
                return clean[:-sep_len], clean[-sep_len:]
        return clean[:3], clean[3:]

    def update_quote(
        self,
        venue: str,
        symbol: str,
        bid: float,
        ask: float,
        bid_qty: float = 10.0,
        ask_qty: float = 10.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Ingest real-time top-of-book (BBO) quote for a currency pair."""
        if bid <= 0 or ask <= 0 or ask < bid:
            return

        clean_sym = symbol.upper().replace("/", "-")
        base, quote = self.parse_symbol(clean_sym)
        self._currencies.add(base)
        self._currencies.add(quote)

        key = (venue.lower(), clean_sym)
        self._quotes[key] = {
            "venue": venue.lower(),
            "symbol": clean_sym,
            "base": base,
            "quote": quote,
            "bid": bid,
            "ask": ask,
            "bid_qty": bid_qty,
            "ask_qty": ask_qty,
            "timestamp": timestamp or datetime.now(timezone.utc),
        }

    def _build_directed_edges(self) -> List[RateGraphEdge]:
        """Construct all directed currency exchange edges with fee and slippage discounting."""
        edges: List[RateGraphEdge] = []

        for (venue, sym), q in self._quotes.items():
            base = q["base"]
            quote = q["quote"]
            fee_bps = self.get_fee_bps(venue)
            fee_rate = fee_bps / 10000.0
            slip_rate = self.default_slippage_bps / 10000.0
            friction = 1.0 - (fee_rate + slip_rate)

            # Edge 1: Quote -> Base (BUYing Base with Quote at Ask price)
            # 1 Quote buys (1.0 / Ask) Base
            ask_price = q["ask"]
            if ask_price > 0:
                rate_buy = 1.0 / ask_price
                eff_rate_buy = rate_buy * friction
                weight_buy = -math.log(max(1e-8, eff_rate_buy))
                edges.append(RateGraphEdge(
                    from_asset=quote,
                    to_asset=base,
                    symbol=sym,
                    venue=venue,
                    rate=rate_buy,
                    effective_rate=eff_rate_buy,
                    weight=weight_buy,
                    action="BUY",
                    fee_bps=fee_bps,
                ))

            # Edge 2: Base -> Quote (SELLing Base for Quote at Bid price)
            # 1 Base sells for Bid Quote
            bid_price = q["bid"]
            if bid_price > 0:
                rate_sell = bid_price
                eff_rate_sell = rate_sell * friction
                weight_sell = -math.log(max(1e-8, eff_rate_sell))
                edges.append(RateGraphEdge(
                    from_asset=base,
                    to_asset=quote,
                    symbol=sym,
                    venue=venue,
                    rate=rate_sell,
                    effective_rate=eff_rate_sell,
                    weight=weight_sell,
                    action="SELL",
                fee_bps=fee_bps,
                ))

        return edges

    def get_rate_graph(self) -> RateGraphData:
        """Export current directed rate graph nodes and edges."""
        edges = self._build_directed_edges()
        nodes = sorted(list(self._currencies))
        return RateGraphData(nodes=nodes, edges=edges)

    def scan_triangular_arbitrage(
        self,
        anchor_asset: str = "USD",
        venues: Optional[List[str]] = None,
    ) -> TriangularScanReport:
        """Scan all currency triangles starting and ending at anchor_asset for profitable loops."""
        edges = self._build_directed_edges()
        target_venues = [v.lower() for v in venues] if venues else None

        if target_venues:
            edges = [e for e in edges if e.venue in target_venues]

        # Build adjacency mapping: from_asset -> list of edges
        adj: Dict[str, List[RateGraphEdge]] = defaultdict(list)
        for e in edges:
            adj[e.from_asset].append(e)

        anchor = anchor_asset.upper()
        opportunities: List[TriangularArbitrageOpportunity] = []
        triangles_evaluated = 0

        # Traverse 3-step cyclic paths: anchor -> B -> C -> anchor
        for e1 in adj[anchor]:
            b = e1.to_asset
            if b == anchor:
                continue

            for e2 in adj[b]:
                c = e2.to_asset
                if c == anchor or c == b:
                    continue

                for e3 in adj[c]:
                    if e3.to_asset != anchor:
                        continue

                    triangles_evaluated += 1
                    # Evaluate loop: anchor -> b -> c -> anchor
                    gross_prod = e1.rate * e2.rate * e3.rate
                    net_prod = e1.effective_rate * e2.effective_rate * e3.effective_rate

                    net_return_bps = (net_prod - 1.0) * 10000.0

                    if net_return_bps >= self.min_net_profit_bps:
                        # Construct legs and calculate executable depth
                        q1 = self._quotes.get((e1.venue, e1.symbol))
                        q2 = self._quotes.get((e2.venue, e2.symbol))
                        q3 = self._quotes.get((e3.venue, e3.symbol))

                        # Depth calculation: convert max inputs to anchor currency units
                        depth1 = (q1["ask_qty"] * q1["ask"]) if e1.action == "BUY" else q1["bid_qty"]
                        depth2 = (q2["ask_qty"] * q2["ask"]) if e2.action == "BUY" else q2["bid_qty"]
                        depth3 = (q3["ask_qty"] * q3["ask"]) if e3.action == "BUY" else q3["bid_qty"]

                        # Capacity in anchor currency
                        cap1 = depth1
                        cap2 = depth2 / max(1e-6, e1.effective_rate)
                        cap3 = depth3 / max(1e-6, e1.effective_rate * e2.effective_rate)

                        max_cap_usd = min(self.max_capital_cap_usd, max(10.0, min(cap1, cap2, cap3)))
                        expected_profit_usd = max_cap_usd * (net_prod - 1.0)

                        venues_in_loop = list(dict.fromkeys([e1.venue, e2.venue, e3.venue]))
                        is_cross = len(venues_in_loop) > 1

                        total_fees = e1.fee_bps + e2.fee_bps + e3.fee_bps
                        total_slip = self.default_slippage_bps * 3.0

                        urgency = "CRITICAL" if net_return_bps >= 25.0 else "HIGH"

                        legs = [
                            TriangularLeg(
                                leg_index=1,
                                from_asset=anchor,
                                to_asset=b,
                                symbol=e1.symbol,
                                venue=e1.venue,
                                action=e1.action,
                                rate=round(e1.rate, 6),
                                quote_price=q1["ask"] if e1.action == "BUY" else q1["bid"],
                                max_input_qty=round(depth1, 4),
                                fee_bps=e1.fee_bps,
                                slippage_bps=self.default_slippage_bps,
                            ),
                            TriangularLeg(
                                leg_index=2,
                                from_asset=b,
                                to_asset=c,
                                symbol=e2.symbol,
                                venue=e2.venue,
                                action=e2.action,
                                rate=round(e2.rate, 6),
                                quote_price=q2["ask"] if e2.action == "BUY" else q2["bid"],
                                max_input_qty=round(depth2, 4),
                                fee_bps=e2.fee_bps,
                                slippage_bps=self.default_slippage_bps,
                            ),
                            TriangularLeg(
                                leg_index=3,
                                from_asset=c,
                                to_asset=anchor,
                                symbol=e3.symbol,
                                venue=e3.venue,
                                action=e3.action,
                                rate=round(e3.rate, 6),
                                quote_price=q3["ask"] if e3.action == "BUY" else q3["bid"],
                                max_input_qty=round(depth3, 4),
                                fee_bps=e3.fee_bps,
                                slippage_bps=self.default_slippage_bps,
                            ),
                        ]

                        opp = TriangularArbitrageOpportunity(
                            triangle_path=[anchor, b, c, anchor],
                            venues_involved=venues_in_loop,
                            is_cross_venue=is_cross,
                            gross_product=round(gross_prod, 6),
                            net_product=round(net_prod, 6),
                            gross_return_pct=round((gross_prod - 1.0) * 100.0, 4),
                            net_return_pct=round((net_prod - 1.0) * 100.0, 4),
                            net_return_bps=round(net_return_bps, 2),
                            total_fees_bps=round(total_fees, 2),
                            total_slippage_bps=round(total_slip, 2),
                            max_executable_capital_usd=round(max_cap_usd, 2),
                            expected_net_profit_usd=round(expected_profit_usd, 2),
                            execution_urgency=urgency,
                            legs=legs,
                        )
                        opportunities.append(opp)

        # Sort opportunities by descending net profit in bps
        opportunities.sort(key=lambda o: o.net_return_bps, reverse=True)
        best = opportunities[0] if opportunities else None

        return TriangularScanReport(
            total_currency_pairs_active=len(self._quotes),
            total_triangles_evaluated=triangles_evaluated,
            opportunities_count=len(opportunities),
            best_opportunity=best,
            opportunities=opportunities,
        )

    def clear_quotes(self) -> None:
        """Purge all cached quotes and currencies."""
        self._quotes.clear()
        self._currencies.clear()
