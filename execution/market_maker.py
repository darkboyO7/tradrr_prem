"""Trading Intelligence Platform - Dynamic Market Maker (DMM) & Automated Quoting Surface Engine.

Implements the Avellaneda-Stoikov high-frequency market making model with
reservation price skewing, multi-level quoting ladders, inventory risk limits,
and adverse selection defense against toxic Order Flow Imbalance (OFI).
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("market_maker")


class InventoryState(str, Enum):
    """Institutional classification of market maker inventory exposure."""

    BALANCED = "BALANCED"
    LONG_SKEWED = "LONG_SKEWED"
    SHORT_SKEWED = "SHORT_SKEWED"
    LIMIT_BREACHED = "LIMIT_BREACHED"


class DMMConfig(BaseModel):
    """Avellaneda-Stoikov parameters and institutional risk constraints."""

    gamma: float = 0.005              # Inventory risk aversion parameter (gamma > 0)
    kappa: float = 1.5                # Order book liquidity density / fill intensity
    time_horizon_days: float = 1.0     # Quoting time horizon (T - t) in days
    max_inventory_units: float = 5.0  # Absolute max inventory ceiling before one-sided halting
    target_inventory: float = 0.0     # Target inventory position (typically 0 for neutral MM)
    quote_levels: int = 3             # Number of laddered quoting tiers
    grid_spacing_bps: float = 4.0     # Spread between ladder levels in basis points
    base_quote_size: float = 0.5      # Default quote order size in base asset units
    min_spread_bps: float = 2.0       # Minimum allowed half-spread to guarantee positive fee capture
    ofi_sensitivity: float = 0.3      # Adverse selection widening factor per unit of OFI


class DMMQuoteLevel(BaseModel):
    """Single tier of two-sided market making quotes."""

    level: int
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    bid_spread_bps: float
    ask_spread_bps: float


class DMMQuoteSurface(BaseModel):
    """Complete multi-tier quoting grid with inventory and adverse selection diagnostics."""

    symbol: str
    mid_price: float
    micro_price: float
    reservation_price: float
    inventory_units: float
    target_inventory_units: float
    inventory_skew_bps: float
    optimal_total_spread_bps: float
    ofi_adverse_selection_bps: float
    inventory_state: InventoryState
    bids_active: bool
    asks_active: bool
    quotes: List[DMMQuoteLevel] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DynamicMarketMakerEngine:
    """Computes inventory-skewed quoting ladders and manages market making risk."""

    def __init__(self, config: Optional[DMMConfig] = None) -> None:
        self.config = config or DMMConfig()
        self._inventories: Dict[str, float] = {}  # {symbol: inventory_units}
        self._realized_pnls: Dict[str, float] = {}

    def get_inventory(self, symbol: str) -> float:
        """Retrieve current net inventory position."""
        return self._inventories.get(symbol, 0.0)

    def set_inventory(self, symbol: str, inventory: float) -> None:
        """Manually override or synchronize inventory position."""
        self._inventories[symbol] = inventory

    def record_fill(self, symbol: str, side: str, quantity: float, price: float) -> float:
        """Record trade execution fill and update inventory balance."""
        curr_inv = self._inventories.get(symbol, 0.0)
        signed_qty = quantity if side.upper() == "BUY" else -quantity
        new_inv = curr_inv + signed_qty
        self._inventories[symbol] = round(new_inv, 6)

        logger.info(
            f"DMM fill recorded on {symbol}: {side} {quantity:.4f} @ ${price:,.2f}. Inventory: {curr_inv:.4f} -> {new_inv:.4f}"
        )
        return self._inventories[symbol]

    def calculate_quotes(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        bid_volume: float = 1.0,
        ask_volume: float = 1.0,
        volatility_annualized: float = 0.50,
        order_flow_imbalance: float = 0.0,
    ) -> DMMQuoteSurface:
        """Generate Avellaneda-Stoikov quoting ladder with OFI skewing and inventory protection."""
        q = self.get_inventory(symbol)
        q_target = self.config.target_inventory
        delta_q = q - q_target

        # 1. Reference Micro-Price (Volume-Weighted Midpoint)
        total_vol = bid_volume + ask_volume
        if total_vol > 0:
            micro_price = (best_bid * ask_volume + best_ask * bid_volume) / total_vol
        else:
            micro_price = (best_bid + best_ask) / 2.0

        mid_price = (best_bid + best_ask) / 2.0

        # Convert annualized volatility to daily/horizon volatility
        t_horizon = max(1e-4, self.config.time_horizon_days / 365.0)
        sigma = volatility_annualized
        gamma = self.config.gamma
        kappa = self.config.kappa

        # 2. Avellaneda-Stoikov Reservation Price r(s, q, t)
        # r = s - (q - q*) * gamma * sigma^2 * (T - t)
        inventory_penalty_usd = delta_q * gamma * (sigma ** 2) * t_horizon * mid_price
        reservation_price = mid_price - inventory_penalty_usd
        inventory_skew_bps = (inventory_penalty_usd / mid_price) * 10000.0

        # 3. Optimal Half-Spread S*(s, q, t)
        # S* = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
        risk_term = gamma * (sigma ** 2) * t_horizon
        liquidity_term = (2.0 / gamma) * math.log(1.0 + gamma / max(1e-5, kappa))
        base_spread_pct = (risk_term + liquidity_term)

        # Adverse Selection / Toxic Flow penalty from OFI
        ofi_penalty_pct = abs(order_flow_imbalance) * self.config.ofi_sensitivity * 0.001
        ofi_penalty_bps = ofi_penalty_pct * 10000.0

        total_half_spread_pct = max(
            self.config.min_spread_bps / 10000.0,
            (base_spread_pct / 2.0) + ofi_penalty_pct,
        )
        total_spread_bps = total_half_spread_pct * 2.0 * 10000.0

        # 4. Inventory Circuit Breakers
        max_inv = self.config.max_inventory_units
        bids_active = True
        asks_active = True

        if abs(q) >= max_inv:
            inv_state = InventoryState.LIMIT_BREACHED
            if q >= max_inv:
                # Excess long -> halt buying, only quote aggressive sells to offload
                bids_active = False
            else:
                # Excess short -> halt selling, only quote aggressive buys
                asks_active = False
        elif abs(q) > (max_inv * 0.5):
            inv_state = InventoryState.LONG_SKEWED if q > 0 else InventoryState.SHORT_SKEWED
        else:
            inv_state = InventoryState.BALANCED

        # 5. Multi-Level Ladder Construction
        quotes: List[DMMQuoteLevel] = []
        base_bid = reservation_price * (1.0 - total_half_spread_pct)
        base_ask = reservation_price * (1.0 + total_half_spread_pct)

        # Ensure quotes don't cross the market
        base_bid = min(base_bid, best_bid)
        base_ask = max(base_ask, best_ask)

        # If OFI is positive (buying surge), widen ask further; if negative (selling surge), widen bid
        if order_flow_imbalance > 0:
            base_ask += mid_price * ofi_penalty_pct
        elif order_flow_imbalance < 0:
            base_bid -= mid_price * ofi_penalty_pct

        grid_bps = self.config.grid_spacing_bps

        for lvl in range(1, self.config.quote_levels + 1):
            offset_pct = (lvl - 1) * (grid_bps / 10000.0)

            # Bid price & spread
            lvl_bid = round(base_bid * (1.0 - offset_pct), 2)
            bid_spr_bps = round(((mid_price - lvl_bid) / mid_price) * 10000.0, 2)
            # Size tapering: outer tiers have larger size
            lvl_bid_size = round(self.config.base_quote_size * (1.0 + 0.5 * (lvl - 1)), 4)

            # Ask price & spread
            lvl_ask = round(base_ask * (1.0 + offset_pct), 2)
            ask_spr_bps = round(((lvl_ask - mid_price) / mid_price) * 10000.0, 2)
            lvl_ask_size = round(self.config.base_quote_size * (1.0 + 0.5 * (lvl - 1)), 4)

            # Zero out size if one-sided halt triggered
            effective_bid_size = lvl_bid_size if bids_active else 0.0
            effective_ask_size = lvl_ask_size if asks_active else 0.0

            quotes.append(
                DMMQuoteLevel(
                    level=lvl,
                    bid_price=lvl_bid,
                    bid_size=effective_bid_size,
                    ask_price=lvl_ask,
                    ask_size=effective_ask_size,
                    bid_spread_bps=bid_spr_bps,
                    ask_spread_bps=ask_spr_bps,
                )
            )

        return DMMQuoteSurface(
            symbol=symbol,
            mid_price=round(mid_price, 2),
            micro_price=round(micro_price, 2),
            reservation_price=round(reservation_price, 2),
            inventory_units=round(q, 4),
            target_inventory_units=q_target,
            inventory_skew_bps=round(inventory_skew_bps, 2),
            optimal_total_spread_bps=round(total_spread_bps, 2),
            ofi_adverse_selection_bps=round(ofi_penalty_bps, 2),
            inventory_state=inv_state,
            bids_active=bids_active,
            asks_active=asks_active,
            quotes=quotes,
        )
