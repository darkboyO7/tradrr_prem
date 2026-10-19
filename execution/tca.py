"""Trading Intelligence Platform - Post-Trade Transaction Cost Analysis (TCA) & Slippage Attribution.

Implements institutional post-trade execution quality benchmarking conforming to
SEC Rule 605/606 and MiFID II RTS 27/28 standards. Decomposes Implementation Shortfall
into delay/latency slippage, effective half-spread, market impact, opportunity cost,
and broker/venue fee drag.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
from pydantic import BaseModel, Field
import scipy.stats as st

from core.logging import get_logger

logger = get_logger("tca_engine")


class OrderBenchmarkType(str, Enum):
    """Reference price benchmark for performance attribution."""

    DECISION_PRICE = "DECISION_PRICE"  # Portfolio manager / signal trigger price
    ARRIVAL_PRICE = "ARRIVAL_PRICE"    # BBO midpoint upon order arrival at broker/venue
    TWAP_BENCHMARK = "TWAP_BENCHMARK"  # Time-weighted average price over execution span
    VWAP_BENCHMARK = "VWAP_BENCHMARK"  # Volume-weighted average price over execution span
    CLOSE_PRICE = "CLOSE_PRICE"        # Market closing price


class TradeExecutionRecord(BaseModel):
    """Institutional execution telemetry for post-trade transaction cost analysis."""

    order_id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:8]}")
    symbol: str
    side: str  # "BUY" or "SELL"
    venue: str
    order_quantity: float
    filled_quantity: float
    decision_price: float
    arrival_price: float
    arrival_bid: float
    arrival_ask: float
    execution_price: float
    terminal_price: Optional[float] = None
    decision_time: datetime
    arrival_time: datetime
    execution_time: datetime
    venue_fees_usd: float = 0.0


class ImplementationShortfallDecomposition(BaseModel):
    """Complete mathematical breakdown of execution shortfall and cost drivers."""

    order_id: str
    symbol: str
    side: str
    venue: str
    notional_usd: float
    total_shortfall_usd: float
    total_shortfall_bps: float
    delay_cost_usd: float
    delay_cost_bps: float
    spread_cost_usd: float
    spread_cost_bps: float
    market_impact_usd: float
    market_impact_bps: float
    opportunity_cost_usd: float
    opportunity_cost_bps: float
    fee_drag_usd: float
    fee_drag_bps: float
    latency_ms: float
    price_improvement_usd: float
    is_price_improved: bool


class VenueExecutionScorecard(BaseModel):
    """Institutional broker and exchange execution quality metrics."""

    venue: str
    total_orders: int
    total_volume_usd: float
    fill_rate_pct: float
    average_effective_spread_bps: float
    average_quoted_spread_bps: float
    effective_to_quoted_ratio: float  # EFQ < 1.0 indicates price improvement
    price_improvement_pct: float
    average_latency_ms: float
    average_implementation_shortfall_bps: float


class TCAStatisticalDistribution(BaseModel):
    """Statistical slippage and shortfall distribution profile."""

    count: int
    mean_shortfall_bps: float
    std_shortfall_bps: float
    p50_shortfall_bps: float
    p95_shortfall_bps: float
    p99_shortfall_bps: float
    skewness: float


class TCAPortfolioReport(BaseModel):
    """Aggregated institutional TCA audit report."""

    report_id: str = Field(default_factory=lambda: f"tca_{uuid.uuid4().hex[:8]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_orders_analyzed: int
    total_notional_usd: float
    aggregate_shortfall_usd: float
    aggregate_shortfall_bps: float
    aggregate_delay_cost_bps: float
    aggregate_spread_cost_bps: float
    aggregate_market_impact_bps: float
    aggregate_opportunity_cost_bps: float
    aggregate_fee_drag_bps: float
    venue_scorecards: Dict[str, VenueExecutionScorecard]
    slippage_distribution: TCAStatisticalDistribution
    decompositions: List[ImplementationShortfallDecomposition] = []


class TransactionCostAnalysisEngine:
    """Calculates implementation shortfall, cost attribution, and venue execution quality."""

    def decompose_trade(self, trade: TradeExecutionRecord) -> ImplementationShortfallDecomposition:
        """Decompose a single trade execution into distinct economic cost components."""
        is_buy = trade.side.upper() == "BUY"
        sign = 1.0 if is_buy else -1.0

        q_filled = trade.filled_quantity
        q_unfilled = max(0.0, trade.order_quantity - q_filled)
        notional_filled = q_filled * trade.execution_price

        # 1. Delay Cost: (P_arrival - P_decision) * Q_filled
        delay_diff = (trade.arrival_price - trade.decision_price) * sign
        delay_usd = delay_diff * q_filled
        delay_bps = (delay_diff / trade.decision_price) * 10000.0 if trade.decision_price > 0 else 0.0

        # 2. Spread Cost (Effective Half-Spread): |P_exec - P_mid_arrival| * Q_filled
        mid_arrival = (trade.arrival_bid + trade.arrival_ask) / 2.0
        quoted_half_spread = (trade.arrival_ask - trade.arrival_bid) / 2.0
        effective_spread = (trade.execution_price - mid_arrival) * sign
        spread_usd = max(0.0, effective_spread * q_filled)
        spread_bps = (effective_spread / mid_arrival) * 10000.0 if mid_arrival > 0 else 0.0

        # 3. Market Impact: price movement beyond arrival half-spread
        impact_diff = (trade.execution_price - trade.arrival_price) * sign - max(0.0, effective_spread)
        impact_usd = max(0.0, impact_diff * q_filled)
        impact_bps = (impact_diff / trade.arrival_price) * 10000.0 if trade.arrival_price > 0 else 0.0

        # 4. Opportunity Cost on unexecuted balance
        term_price = trade.terminal_price or trade.execution_price
        opp_diff = (term_price - trade.decision_price) * sign
        opportunity_usd = max(0.0, opp_diff * q_unfilled)
        opportunity_bps = (opp_diff / trade.decision_price) * 10000.0 if trade.decision_price > 0 else 0.0

        # 5. Fee Drag
        fee_usd = trade.venue_fees_usd
        fee_bps = (fee_usd / notional_filled) * 10000.0 if notional_filled > 0 else 0.0

        # Total Shortfall: (P_exec - P_decision) * Q_filled + Opp_Cost + Fees
        exec_shortfall_diff = (trade.execution_price - trade.decision_price) * sign
        total_shortfall_usd = (exec_shortfall_diff * q_filled) + opportunity_usd + fee_usd
        total_shortfall_bps = (total_shortfall_usd / notional_filled) * 10000.0 if notional_filled > 0 else 0.0

        # Latency in milliseconds
        latency_ms = max(0.0, (trade.execution_time - trade.decision_time).total_seconds() * 1000.0)

        # Price Improvement: if executed inside quoted half-spread (better than quoted ask for buy, bid for sell)
        quoted_price = trade.arrival_ask if is_buy else trade.arrival_bid
        price_improvement_diff = (quoted_price - trade.execution_price) * sign
        is_pi = price_improvement_diff > 1e-6
        pi_usd = max(0.0, price_improvement_diff * q_filled)

        return ImplementationShortfallDecomposition(
            order_id=trade.order_id,
            symbol=trade.symbol,
            side=trade.side,
            venue=trade.venue,
            notional_usd=round(notional_filled, 2),
            total_shortfall_usd=round(total_shortfall_usd, 2),
            total_shortfall_bps=round(total_shortfall_bps, 2),
            delay_cost_usd=round(delay_usd, 2),
            delay_cost_bps=round(delay_bps, 2),
            spread_cost_usd=round(spread_usd, 2),
            spread_cost_bps=round(spread_bps, 2),
            market_impact_usd=round(impact_usd, 2),
            market_impact_bps=round(impact_bps, 2),
            opportunity_cost_usd=round(opportunity_usd, 2),
            opportunity_cost_bps=round(opportunity_bps, 2),
            fee_drag_usd=round(fee_usd, 2),
            fee_drag_bps=round(fee_bps, 2),
            latency_ms=round(latency_ms, 2),
            price_improvement_usd=round(pi_usd, 2),
            is_price_improved=is_pi,
        )

    def calculate_venue_scorecards(
        self, trades: List[TradeExecutionRecord]
    ) -> Dict[str, VenueExecutionScorecard]:
        """Aggregate trade records into broker/venue execution scorecards."""
        venue_trades: Dict[str, List[TradeExecutionRecord]] = {}
        for t in trades:
            venue_trades.setdefault(t.venue, []).append(t)

        scorecards: Dict[str, VenueExecutionScorecard] = {}

        for venue, t_list in venue_trades.items():
            tot_orders = len(t_list)
            tot_ordered_qty = sum(t.order_quantity for t in t_list)
            tot_filled_qty = sum(t.filled_quantity for t in t_list)
            fill_rate = (tot_filled_qty / tot_ordered_qty) * 100.0 if tot_ordered_qty > 0 else 0.0

            tot_volume = sum(t.filled_quantity * t.execution_price for t in t_list)

            eff_spreads_bps = []
            quoted_spreads_bps = []
            pi_count = 0
            latencies = []
            shortfalls_bps = []

            for t in t_list:
                mid = (t.arrival_bid + t.arrival_ask) / 2.0
                quoted_spr = (t.arrival_ask - t.arrival_bid)
                quoted_spr_bps = (quoted_spr / mid) * 10000.0 if mid > 0 else 0.0
                eff_spr = 2.0 * abs(t.execution_price - mid)
                eff_spr_bps = (eff_spr / mid) * 10000.0 if mid > 0 else 0.0

                eff_spreads_bps.append(eff_spr_bps)
                quoted_spreads_bps.append(quoted_spr_bps)

                is_buy = t.side.upper() == "BUY"
                quoted_px = t.arrival_ask if is_buy else t.arrival_bid
                if (quoted_px - t.execution_price) * (1.0 if is_buy else -1.0) > 1e-6:
                    pi_count += 1

                lat_ms = (t.execution_time - t.decision_time).total_seconds() * 1000.0
                latencies.append(lat_ms)

                sign = 1.0 if is_buy else -1.0
                sf_bps = ((t.execution_price - t.decision_price) * sign / t.decision_price) * 10000.0
                shortfalls_bps.append(sf_bps)

            avg_eff_spread = float(np.mean(eff_spreads_bps)) if eff_spreads_bps else 0.0
            avg_quoted_spread = float(np.mean(quoted_spreads_bps)) if quoted_spreads_bps else 0.0
            efq = (avg_eff_spread / avg_quoted_spread) if avg_quoted_spread > 0 else 1.0
            pi_pct = (pi_count / tot_orders) * 100.0 if tot_orders > 0 else 0.0
            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            avg_sf = float(np.mean(shortfalls_bps)) if shortfalls_bps else 0.0

            scorecards[venue] = VenueExecutionScorecard(
                venue=venue,
                total_orders=tot_orders,
                total_volume_usd=round(tot_volume, 2),
                fill_rate_pct=round(fill_rate, 2),
                average_effective_spread_bps=round(avg_eff_spread, 2),
                average_quoted_spread_bps=round(avg_quoted_spread, 2),
                effective_to_quoted_ratio=round(efq, 4),
                price_improvement_pct=round(pi_pct, 2),
                average_latency_ms=round(avg_lat, 2),
                average_implementation_shortfall_bps=round(avg_sf, 2),
            )

        return scorecards

    def calculate_slippage_distribution(
        self, shortfalls_bps: List[float]
    ) -> TCAStatisticalDistribution:
        """Compute higher-order statistical moments and percentiles of slippage distribution."""
        if not shortfalls_bps:
            return TCAStatisticalDistribution(
                count=0,
                mean_shortfall_bps=0.0,
                std_shortfall_bps=0.0,
                p50_shortfall_bps=0.0,
                p95_shortfall_bps=0.0,
                p99_shortfall_bps=0.0,
                skewness=0.0,
            )

        arr = np.array(shortfalls_bps)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        skew_val = float(st.skew(arr)) if len(arr) > 2 else 0.0

        return TCAStatisticalDistribution(
            count=len(arr),
            mean_shortfall_bps=round(mean_val, 2),
            std_shortfall_bps=round(std_val, 2),
            p50_shortfall_bps=round(p50, 2),
            p95_shortfall_bps=round(p95, 2),
            p99_shortfall_bps=round(p99, 2),
            skewness=round(skew_val, 4),
        )

    def generate_portfolio_tca_report(
        self, trades: List[TradeExecutionRecord]
    ) -> TCAPortfolioReport:
        """Generate comprehensive portfolio-level TCA audit report."""
        if not trades:
            empty_dist = self.calculate_slippage_distribution([])
            return TCAPortfolioReport(
                total_orders_analyzed=0,
                total_notional_usd=0.0,
                aggregate_shortfall_usd=0.0,
                aggregate_shortfall_bps=0.0,
                aggregate_delay_cost_bps=0.0,
                aggregate_spread_cost_bps=0.0,
                aggregate_market_impact_bps=0.0,
                aggregate_opportunity_cost_bps=0.0,
                aggregate_fee_drag_bps=0.0,
                venue_scorecards={},
                slippage_distribution=empty_dist,
                decompositions=[],
            )

        decomps = [self.decompose_trade(t) for t in trades]
        total_notional = sum(d.notional_usd for d in decomps)
        total_shortfall_usd = sum(d.total_shortfall_usd for d in decomps)

        # Volume-weighted aggregate bps
        def weighted_bps(bps_field: str) -> float:
            if total_notional <= 0:
                return 0.0
            return sum(getattr(d, bps_field) * d.notional_usd for d in decomps) / total_notional

        scorecards = self.calculate_venue_scorecards(trades)
        shortfalls = [d.total_shortfall_bps for d in decomps]
        dist = self.calculate_slippage_distribution(shortfalls)

        return TCAPortfolioReport(
            total_orders_analyzed=len(trades),
            total_notional_usd=round(total_notional, 2),
            aggregate_shortfall_usd=round(total_shortfall_usd, 2),
            aggregate_shortfall_bps=round(weighted_bps("total_shortfall_bps"), 2),
            aggregate_delay_cost_bps=round(weighted_bps("delay_cost_bps"), 2),
            aggregate_spread_cost_bps=round(weighted_bps("spread_cost_bps"), 2),
            aggregate_market_impact_bps=round(weighted_bps("market_impact_bps"), 2),
            aggregate_opportunity_cost_bps=round(weighted_bps("opportunity_cost_bps"), 2),
            aggregate_fee_drag_bps=round(weighted_bps("fee_drag_bps"), 2),
            venue_scorecards=scorecards,
            slippage_distribution=dist,
            decompositions=decomps,
        )
