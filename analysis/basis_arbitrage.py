"""Trading Intelligence Platform - Cross-Venue Synthetic Funding Rate Arbitrage & Basis Trading Engine.

Calculates annualized cash-and-carry basis spreads, project 8-hour funding cashflows,
identifies cross-venue perpetual funding divergence, and synthesizes delta-neutral
arbitrage opportunities accounting for borrow friction, trading fees, and convergence risk.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("basis_arbitrage")


class BasisTradeType(str, Enum):
    """Institutional classification of delta-neutral basis/funding carry structures."""

    CASH_AND_CARRY = "CASH_AND_CARRY"                # Long Spot + Short Perpetual/Future
    REVERSE_CASH_AND_CARRY = "REVERSE_CASH_AND_CARRY" # Short Spot + Long Perpetual/Future
    PERP_PERP_FUNDING_ARB = "PERP_PERP_FUNDING_ARB"   # Long Perp (Low/Neg Funding) + Short Perp (High Funding)


class PerpetualFundingRate(BaseModel):
    """Real-time funding rate telemetry for a perpetual futures contract."""

    venue: str
    symbol: str
    rate_per_interval: float  # e.g., 0.0001 = 0.01% per interval
    interval_hours: int = 8
    mark_price: float
    index_price: float
    next_funding_time: Optional[datetime] = None
    predicted_next_rate: Optional[float] = None

    @property
    def annualized_rate_pct(self) -> float:
        """Annualized funding yield in percent (APR)."""
        intervals_per_year = (24.0 / max(1, self.interval_hours)) * 365.0
        return self.rate_per_interval * intervals_per_year * 100.0


class SpotQuote(BaseModel):
    """Current top-of-book spot price quotation."""

    venue: str
    symbol: str
    bid: float
    ask: float
    mid: float


class BasisOpportunity(BaseModel):
    """Actionable delta-neutral funding rate or basis arbitrage opportunity."""

    opportunity_id: str = Field(default_factory=lambda: f"opp_basis_{uuid.uuid4().hex[:8]}")
    strategy_type: BasisTradeType
    underlying_asset: str
    long_leg_venue: str
    long_leg_symbol: str
    long_leg_price: float
    short_leg_venue: str
    short_leg_symbol: str
    short_leg_price: float
    basis_spread_bps: float
    annualized_basis_yield_pct: float
    net_funding_apr_pct: float
    spot_borrow_apr_pct: float
    trading_fee_bps: float
    estimated_net_apr_pct: float
    recommended_leverage: float
    liquidation_buffer_pct: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BasisPosition(BaseModel):
    """Active delta-neutral basis arbitrage trade tracking."""

    position_id: str = Field(default_factory=lambda: f"pos_basis_{uuid.uuid4().hex[:8]}")
    opportunity: BasisOpportunity
    quantity: float
    entry_basis_bps: float
    current_basis_bps: float
    total_capital_deployed_usd: float
    cumulative_funding_collected_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    realized_funding_yield_pct: float = 0.0
    is_active: bool = True
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BasisArbitrageScanner:
    """Scans and models cross-venue cash-and-carry and perpetual funding rate arbitrage."""

    def __init__(
        self,
        default_taker_fee_bps: float = 5.0,  # 5 bps per leg
        default_maker_fee_bps: float = 2.0,  # 2 bps per leg
        default_borrow_rate_apr: float = 3.0, # 3% annual cost to borrow spot
    ) -> None:
        self.default_taker_fee_bps = default_taker_fee_bps
        self.default_maker_fee_bps = default_maker_fee_bps
        self.default_borrow_rate_apr = default_borrow_rate_apr

        self._spot_quotes: Dict[str, Dict[str, SpotQuote]] = {}  # {underlying: {venue: SpotQuote}}
        self._perp_rates: Dict[str, Dict[str, PerpetualFundingRate]] = {}  # {underlying: {venue: PerpetualFundingRate}}

    def register_spot_quote(
        self, underlying: str, venue: str, symbol: str, bid: float, ask: float
    ) -> None:
        """Register live spot quote for an asset."""
        if underlying not in self._spot_quotes:
            self._spot_quotes[underlying] = {}
        mid = (bid + ask) / 2.0
        self._spot_quotes[underlying][venue] = SpotQuote(
            venue=venue, symbol=symbol, bid=bid, ask=ask, mid=mid
        )

    def register_perp_rate(
        self,
        underlying: str,
        venue: str,
        symbol: str,
        rate_per_interval: float,
        mark_price: float,
        index_price: float,
        interval_hours: int = 8,
        next_funding_time: Optional[datetime] = None,
        predicted_next_rate: Optional[float] = None,
    ) -> None:
        """Register perpetual funding rate and mark price for a venue."""
        if underlying not in self._perp_rates:
            self._perp_rates[underlying] = {}
        self._perp_rates[underlying][venue] = PerpetualFundingRate(
            venue=venue,
            symbol=symbol,
            rate_per_interval=rate_per_interval,
            interval_hours=interval_hours,
            mark_price=mark_price,
            index_price=index_price,
            next_funding_time=next_funding_time,
            predicted_next_rate=predicted_next_rate,
        )

    def scan_opportunities(
        self, min_net_apr_pct: float = 4.0, target_holding_days: float = 30.0
    ) -> List[BasisOpportunity]:
        """Scan all registered assets and venues for actionable basis carry opportunities."""
        opportunities: List[BasisOpportunity] = []

        all_underlyings = set(self._spot_quotes.keys()) | set(self._perp_rates.keys())

        for asset in all_underlyings:
            spots = self._spot_quotes.get(asset, {})
            perps = self._perp_rates.get(asset, {})

            # 1. Cash-and-Carry (Long Spot + Short Perp)
            for s_venue, s_quote in spots.items():
                for p_venue, p_quote in perps.items():
                    spot_buy = s_quote.ask
                    perp_sell = p_quote.mark_price

                    basis_spread = (perp_sell - spot_buy) / spot_buy
                    basis_bps = basis_spread * 10000.0

                    annualized_basis = (basis_spread * (365.0 / target_holding_days)) * 100.0
                    funding_apr = p_quote.annualized_rate_pct  # Short collects positive funding

                    # Net APR calculation
                    total_fees_bps = 2.0 * (self.default_taker_fee_bps + self.default_maker_fee_bps)
                    fee_drag_apr = (total_fees_bps / 10000.0) * (365.0 / target_holding_days) * 100.0

                    net_apr = annualized_basis + funding_apr - fee_drag_apr

                    if net_apr >= min_net_apr_pct and funding_apr > 0.0:
                        opp = BasisOpportunity(
                            strategy_type=BasisTradeType.CASH_AND_CARRY,
                            underlying_asset=asset,
                            long_leg_venue=s_venue,
                            long_leg_symbol=s_quote.symbol,
                            long_leg_price=spot_buy,
                            short_leg_venue=p_venue,
                            short_leg_symbol=p_quote.symbol,
                            short_leg_price=perp_sell,
                            basis_spread_bps=round(basis_bps, 2),
                            annualized_basis_yield_pct=round(annualized_basis, 2),
                            net_funding_apr_pct=round(funding_apr, 2),
                            spot_borrow_apr_pct=0.0,
                            trading_fee_bps=round(total_fees_bps, 1),
                            estimated_net_apr_pct=round(net_apr, 2),
                            recommended_leverage=2.0,
                            liquidation_buffer_pct=35.0,
                        )
                        opportunities.append(opp)

            # 2. Reverse Cash-and-Carry (Short Spot + Long Perp)
            for s_venue, s_quote in spots.items():
                for p_venue, p_quote in perps.items():
                    spot_sell = s_quote.bid
                    perp_buy = p_quote.mark_price

                    # Profit if perp is traded at discount or funding is deeply negative (longs get paid)
                    basis_spread = (spot_sell - perp_buy) / spot_sell
                    basis_bps = basis_spread * 10000.0

                    annualized_basis = (basis_spread * (365.0 / target_holding_days)) * 100.0
                    funding_apr = -p_quote.annualized_rate_pct  # Long collects if funding is negative

                    total_fees_bps = 2.0 * (self.default_taker_fee_bps + self.default_maker_fee_bps)
                    fee_drag_apr = (total_fees_bps / 10000.0) * (365.0 / target_holding_days) * 100.0

                    net_apr = annualized_basis + funding_apr - self.default_borrow_rate_apr - fee_drag_apr

                    if net_apr >= min_net_apr_pct and funding_apr > 0.0:
                        opp = BasisOpportunity(
                            strategy_type=BasisTradeType.REVERSE_CASH_AND_CARRY,
                            underlying_asset=asset,
                            long_leg_venue=p_venue,
                            long_leg_symbol=p_quote.symbol,
                            long_leg_price=perp_buy,
                            short_leg_venue=s_venue,
                            short_leg_symbol=s_quote.symbol,
                            short_leg_price=spot_sell,
                            basis_spread_bps=round(basis_bps, 2),
                            annualized_basis_yield_pct=round(annualized_basis, 2),
                            net_funding_apr_pct=round(funding_apr, 2),
                            spot_borrow_apr_pct=self.default_borrow_rate_apr,
                            trading_fee_bps=round(total_fees_bps, 1),
                            estimated_net_apr_pct=round(net_apr, 2),
                            recommended_leverage=2.0,
                            liquidation_buffer_pct=30.0,
                        )
                        opportunities.append(opp)

            # 3. Cross-Venue Perpetual-Perpetual Funding Arbitrage
            perp_venues = list(perps.keys())
            for i in range(len(perp_venues)):
                for j in range(i + 1, len(perp_venues)):
                    v_a = perp_venues[i]
                    v_b = perp_venues[j]
                    p_a = perps[v_a]
                    p_b = perps[v_b]

                    # Long lower funding venue, Short higher funding venue
                    if p_a.annualized_rate_pct < p_b.annualized_rate_pct:
                        v_long, p_long = v_a, p_a
                        v_short, p_short = v_b, p_b
                    else:
                        v_long, p_long = v_b, p_b
                        v_short, p_short = v_a, p_a

                    funding_diff_apr = p_short.annualized_rate_pct - p_long.annualized_rate_pct
                    price_basis_spread = (p_short.mark_price - p_long.mark_price) / p_long.mark_price
                    basis_bps = price_basis_spread * 10000.0

                    total_fees_bps = 2.0 * (self.default_taker_fee_bps + self.default_maker_fee_bps)
                    fee_drag_apr = (total_fees_bps / 10000.0) * (365.0 / target_holding_days) * 100.0
                    net_apr = funding_diff_apr - fee_drag_apr

                    if net_apr >= min_net_apr_pct:
                        opp = BasisOpportunity(
                            strategy_type=BasisTradeType.PERP_PERP_FUNDING_ARB,
                            underlying_asset=asset,
                            long_leg_venue=v_long,
                            long_leg_symbol=p_long.symbol,
                            long_leg_price=p_long.mark_price,
                            short_leg_venue=v_short,
                            short_leg_symbol=p_short.symbol,
                            short_leg_price=p_short.mark_price,
                            basis_spread_bps=round(basis_bps, 2),
                            annualized_basis_yield_pct=0.0,
                            net_funding_apr_pct=round(funding_diff_apr, 2),
                            spot_borrow_apr_pct=0.0,
                            trading_fee_bps=round(total_fees_bps, 1),
                            estimated_net_apr_pct=round(net_apr, 2),
                            recommended_leverage=3.0,
                            liquidation_buffer_pct=25.0,
                        )
                        opportunities.append(opp)

        # Sort highest estimated net APR first
        opportunities.sort(key=lambda x: x.estimated_net_apr_pct, reverse=True)
        return opportunities


class BasisPositionTracker:
    """Manages active delta-neutral carry trades, cashflows, and convergence tracking."""

    def __init__(self) -> None:
        self._positions: Dict[str, BasisPosition] = {}

    def open_position(self, opportunity: BasisOpportunity, quantity: float) -> BasisPosition:
        """Initialize and record a new delta-neutral carry trade."""
        deployed_usd = quantity * (opportunity.long_leg_price + opportunity.short_leg_price)
        pos = BasisPosition(
            opportunity=opportunity,
            quantity=quantity,
            entry_basis_bps=opportunity.basis_spread_bps,
            current_basis_bps=opportunity.basis_spread_bps,
            total_capital_deployed_usd=round(deployed_usd, 2),
        )
        self._positions[pos.position_id] = pos
        logger.info(
            f"Opened basis trade {pos.position_id}: {opportunity.strategy_type.value} on {opportunity.underlying_asset} (${deployed_usd:,.2f})"
        )
        return pos

    def accrue_funding_payment(
        self, position_id: str, funding_cashflow_usd: float
    ) -> Optional[BasisPosition]:
        """Record an 8-hour / 1-hour periodic funding settlement payment."""
        pos = self._positions.get(position_id)
        if not pos:
            return None
        pos.cumulative_funding_collected_usd += funding_cashflow_usd
        if pos.total_capital_deployed_usd > 0:
            pos.realized_funding_yield_pct = round(
                (pos.cumulative_funding_collected_usd / pos.total_capital_deployed_usd) * 100.0, 4
            )
        return pos

    def update_mark(
        self, position_id: str, current_long_price: float, current_short_price: float
    ) -> Optional[BasisPosition]:
        """Update mark-to-market valuations and convergence basis."""
        pos = self._positions.get(position_id)
        if not pos:
            return None

        current_basis = (current_short_price - current_long_price) / current_long_price
        pos.current_basis_bps = round(current_basis * 10000.0, 2)

        # Delta-neutral MTM: long PnL + short PnL
        long_pnl = pos.quantity * (current_long_price - pos.opportunity.long_leg_price)
        short_pnl = pos.quantity * (pos.opportunity.short_leg_price - current_short_price)
        pos.unrealized_pnl_usd = round(long_pnl + short_pnl, 2)
        return pos

    def get_position(self, position_id: str) -> Optional[BasisPosition]:
        return self._positions.get(position_id)

    def list_active_positions(self) -> List[BasisPosition]:
        return [p for p in self._positions.values() if p.is_active]
