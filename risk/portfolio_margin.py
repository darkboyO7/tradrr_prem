"""Trading Intelligence Platform - Institutional Portfolio Margin & Cross-Collateralization Engine.

Calculates multi-asset haircut discounts, maintenance/initial margin requirements,
netting credits for hedged positions, distance to margin call / liquidation triggers,
and multi-subaccount risk budget segregation.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("portfolio_margin")


class MarginStatusLevel(str, Enum):
    """Institutional classification of account margin health."""

    HEALTHY = "HEALTHY"                     # Usage < 60%
    WARNING = "WARNING"                     # 60% <= Usage < 80%
    MARGIN_CALL = "MARGIN_CALL"             # 80% <= Usage < 95%
    LIQUIDATION_BREACH = "LIQUIDATION_BREACH" # Usage >= 95%


class CollateralAsset(BaseModel):
    """Collateral asset with regulatory haircut valuation."""

    asset: str
    quantity: float
    mark_price: float
    haircut_pct: float = 0.0

    @property
    def nominal_value_usd(self) -> float:
        return self.quantity * self.mark_price

    @property
    def collateral_value_usd(self) -> float:
        return self.nominal_value_usd * (1.0 - self.haircut_pct)


class MarginPosition(BaseModel):
    """Derivative or spot leveraged position for margin requirement calculation."""

    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    mark_price: float
    maintenance_margin_rate: float = 0.05
    initial_margin_rate: float = 0.10
    unrealized_pnl: float = 0.0

    @property
    def notional_usd(self) -> float:
        return abs(self.quantity * self.mark_price)

    @property
    def maintenance_margin_usd(self) -> float:
        return self.notional_usd * self.maintenance_margin_rate

    @property
    def initial_margin_usd(self) -> float:
        return self.notional_usd * self.initial_margin_rate


class SubAccountBook(BaseModel):
    """Segregated strategy sub-account book with isolated PnL and risk budget."""

    subaccount_id: str
    strategy_name: str
    cash: float
    allocated_risk_budget_usd: float
    positions: List[MarginPosition] = []


class PortfolioMarginReport(BaseModel):
    """Comprehensive portfolio margin and cross-collateralization audit report."""

    total_nominal_equity_usd: float
    total_collateral_value_usd: float
    total_initial_margin_usd: float
    total_maintenance_margin_usd: float
    netting_benefit_usd: float
    net_maintenance_margin_usd: float
    margin_usage_pct: float
    margin_buffer_pct: float
    free_margin_usd: float
    status: MarginStatusLevel
    action_required: str
    subaccount_summaries: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PortfolioMarginEngine:
    """Calculates cross-margined portfolio requirements, haircuts, and liquidation buffers."""

    DEFAULT_HAIRCUTS = {
        "USD": 0.0,
        "USDT": 0.01,
        "USDC": 0.01,
        "BTC": 0.10,
        "ETH": 0.15,
        "SOL": 0.25,
        "DEFAULT": 0.35,
    }

    DEFAULT_MM_RATES = {
        "BTC-USD": 0.05,
        "ETH-USD": 0.05,
        "SOL-USD": 0.10,
        "DEFAULT": 0.15,
    }

    def __init__(
        self,
        custom_haircuts: Optional[Dict[str, float]] = None,
        custom_mm_rates: Optional[Dict[str, float]] = None,
        hedged_netting_discount: float = 0.60,  # 60% margin relief on hedged long/short positions
    ) -> None:
        self.haircuts = {**self.DEFAULT_HAIRCUTS, **(custom_haircuts or {})}
        self.mm_rates = {**self.DEFAULT_MM_RATES, **(custom_mm_rates or {})}
        self.hedged_netting_discount = hedged_netting_discount
        self.subaccounts: Dict[str, SubAccountBook] = {}

    def get_haircut(self, asset: str) -> float:
        """Retrieve haircut percentage for a collateral asset."""
        return self.haircuts.get(asset.upper(), self.haircuts["DEFAULT"])

    def get_mm_rate(self, symbol: str) -> float:
        """Retrieve maintenance margin rate for a trading symbol."""
        return self.mm_rates.get(symbol.upper(), self.mm_rates["DEFAULT"])

    def register_subaccount(self, subaccount: SubAccountBook) -> None:
        """Register a segregated strategy book."""
        self.subaccounts[subaccount.subaccount_id] = subaccount

    def evaluate_margin(
        self,
        collateral_assets: List[CollateralAsset],
        positions: List[MarginPosition],
        cash_balance: float = 0.0,
    ) -> PortfolioMarginReport:
        """Calculate complete master-account portfolio margin with haircut discounting and netting."""
        # 1. Total Collateral Value (with haircuts)
        total_nominal_equity = cash_balance
        total_collateral_value = cash_balance  # Cash has 0% haircut

        for ca in collateral_assets:
            total_nominal_equity += ca.nominal_value_usd
            total_collateral_value += ca.collateral_value_usd

        # Factor in unrealized PnL from active positions
        total_upnl = sum(p.unrealized_pnl for p in positions)
        total_nominal_equity += total_upnl
        total_collateral_value += total_upnl

        # 2. Margin Requirements
        gross_im = sum(p.initial_margin_usd for p in positions)
        gross_mm = sum(p.maintenance_margin_usd for p in positions)

        # 3. Netting benefit: Calculate offsetting long/short exposure per symbol
        symbol_longs: Dict[str, float] = {}
        symbol_shorts: Dict[str, float] = {}

        for p in positions:
            sym = p.symbol.upper()
            if p.side.upper() == "LONG":
                symbol_longs[sym] = symbol_longs.get(sym, 0.0) + p.notional_usd
            else:
                symbol_shorts[sym] = symbol_shorts.get(sym, 0.0) + p.notional_usd

        netting_benefit = 0.0
        all_symbols = set(symbol_longs.keys()).union(set(symbol_shorts.keys()))
        for sym in all_symbols:
            long_exp = symbol_longs.get(sym, 0.0)
            short_exp = symbol_shorts.get(sym, 0.0)
            hedged_notional = min(long_exp, short_exp)
            if hedged_notional > 0:
                mm_rate = self.get_mm_rate(sym)
                # Apply relief discount to the hedged portion
                relief = hedged_notional * mm_rate * self.hedged_netting_discount
                netting_benefit += relief

        net_mm = max(0.0, gross_mm - netting_benefit)

        # 4. Margin Usage Ratio & Free Margin
        if total_collateral_value > 0:
            usage_ratio = (net_mm / total_collateral_value) * 100.0
            buffer_pct = max(0.0, 100.0 - usage_ratio)
        else:
            usage_ratio = 100.0 if net_mm > 0 else 0.0
            buffer_pct = 0.0

        free_margin = max(0.0, total_collateral_value - net_mm)

        # 5. Status Classification
        if usage_ratio >= 95.0:
            status = MarginStatusLevel.LIQUIDATION_BREACH
            action = "IMMEDIATE_DELEVERAGING_REQUIRED: Account in liquidation breach zone"
        elif usage_ratio >= 80.0:
            status = MarginStatusLevel.MARGIN_CALL
            action = "MARGIN_CALL_ALERT: Deposit collateral or reduce leveraged positions"
        elif usage_ratio >= 60.0:
            status = MarginStatusLevel.WARNING
            action = "ELEVATED_LEVERAGE: Monitor collateral buffer and restrict new risk"
        else:
            status = MarginStatusLevel.HEALTHY
            action = "NORMAL: Account sufficiently collateralized"

        # 6. Subaccount Summaries
        sub_summaries = {}
        for s_id, s_book in self.subaccounts.items():
            s_upnl = sum(p.unrealized_pnl for p in s_book.positions)
            s_notional = sum(p.notional_usd for p in s_book.positions)
            sub_summaries[s_id] = {
                "strategy_name": s_book.strategy_name,
                "cash": s_book.cash,
                "positions_count": len(s_book.positions),
                "total_notional_usd": round(s_notional, 2),
                "unrealized_pnl": round(s_upnl, 2),
                "allocated_budget": s_book.allocated_risk_budget_usd,
            }

        logger.info(
            "Portfolio margin evaluated",
            equity=round(total_nominal_equity, 2),
            collateral=round(total_collateral_value, 2),
            maintenance_margin=round(net_mm, 2),
            usage_pct=round(usage_ratio, 2),
            status=status.value,
        )

        return PortfolioMarginReport(
            total_nominal_equity_usd=round(total_nominal_equity, 2),
            total_collateral_value_usd=round(total_collateral_value, 2),
            total_initial_margin_usd=round(gross_im, 2),
            total_maintenance_margin_usd=round(gross_mm, 2),
            netting_benefit_usd=round(netting_benefit, 2),
            net_maintenance_margin_usd=round(net_mm, 2),
            margin_usage_pct=round(usage_ratio, 2),
            margin_buffer_pct=round(buffer_pct, 2),
            free_margin_usd=round(free_margin, 2),
            status=status,
            action_required=action,
            subaccount_summaries=sub_summaries,
        )
