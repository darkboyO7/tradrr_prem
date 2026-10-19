"""Trading Intelligence Platform - Core Instrument, Market & Symbol Specifications.

Defines asset classes, market types, canonical instrument data structures,
exchange trading session definitions, tick/lot specifications, and fees.
"""

from datetime import UTC, datetime, time
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field


class MarketType(str, Enum):
    """Supported market segments."""
    NSE = "NSE"
    BSE = "BSE"
    FOREX = "FOREX"
    QUOTEX = "QUOTEX"
    CRYPTO = "CRYPTO"


class AssetClass(str, Enum):
    """Institutional asset classifications."""
    EQUITY = "EQUITY"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"
    BINARY_OPTION = "BINARY_OPTION"
    COMMODITY = "COMMODITY"
    INDEX = "INDEX"


class SettlementType(str, Enum):
    """Settlement rules."""
    T_PLUS_0 = "T+0"
    T_PLUS_1 = "T+1"
    T_PLUS_2 = "T+2"
    INSTANT = "INSTANT"


class TradingSession(BaseModel):
    """Market operating schedule and timezone."""
    model_config = ConfigDict(frozen=True)

    market: MarketType
    timezone: str = "UTC"
    start_time: str = "00:00"  # HH:MM local
    end_time: str = "23:59"    # HH:MM local
    trading_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # 0=Monday, 6=Sunday
    is_24_7: bool = False

    def is_open(self, dt: datetime | None = None) -> bool:
        """Check if trading session is currently active given UTC or local time."""
        if self.is_24_7:
            return True

        eval_dt = dt or datetime.now(UTC)
        try:
            tz = ZoneInfo(self.timezone)
            local_dt = eval_dt.astimezone(tz)
        except Exception:
            local_dt = eval_dt

        # Check trading days
        if local_dt.weekday() not in self.trading_days:
            return False

        # Parse times
        start_h, start_m = map(int, self.start_time.split(":"))
        end_h, end_m = map(int, self.end_time.split(":"))
        cur_t = local_dt.time()

        start_t = time(start_h, start_m)
        end_t = time(end_h, end_m)

        if start_t <= end_t:
            return start_t <= cur_t <= end_t
        else:
            # Overnights
            return cur_t >= start_t or cur_t <= end_t


class Instrument(BaseModel):
    """Institutional canonical Instrument model."""
    model_config = ConfigDict(frozen=True)

    symbol: str
    canonical_symbol: str
    market: MarketType
    exchange: str
    asset_class: AssetClass
    base_currency: str
    quote_currency: str
    tick_size: float = Field(default=0.01, gt=0.0)
    lot_size: float = Field(default=1.0, gt=0.0)
    price_precision: int = Field(default=2, ge=0)
    quantity_precision: int = Field(default=2, ge=0)
    contract_multiplier: float = Field(default=1.0, gt=0.0)
    settlement_type: SettlementType = Field(default=SettlementType.T_PLUS_1)
    maker_fee_pct: float = Field(default=0.001)
    taker_fee_pct: float = Field(default=0.002)
    supported_order_types: list[str] = Field(
        default_factory=lambda: ["MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT"]
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def round_price(self, price: float) -> float:
        """Round price to the instrument's tick size and decimal precision."""
        if self.tick_size <= 0:
            return round(price, self.price_precision)
        ticks = round(price / self.tick_size)
        return round(ticks * self.tick_size, self.price_precision)

    def round_quantity(self, quantity: float) -> float:
        """Round quantity to the instrument's lot size and quantity precision."""
        if self.lot_size <= 0:
            return round(quantity, self.quantity_precision)
        lots = round(quantity / self.lot_size)
        qty = lots * self.lot_size
        return round(qty, self.quantity_precision)
