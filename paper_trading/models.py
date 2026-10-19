"""Trading Intelligence Platform - Paper Trading Domain Models.

Represents virtual orders, execution traces, and portfolio positions with strict MTM valuation.
"""

from datetime import UTC, datetime
from enum import Enum

import uuid
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PaperOrder(BaseModel):
    """Virtual simulated order routed to paper trading engine."""

    order_id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:8]}")
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.SUBMITTED
    filled_price: float | None = None
    filled_quantity: float = 0.0
    fee: float = 0.0
    slippage: float = 0.0
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    filled_at: datetime | None = None
    stop_loss: float | None = None
    take_profit: float | None = None


class PaperPosition(BaseModel):
    """Active or closed portfolio position with mark-to-market PnL."""

    position_id: str
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_open: bool = True
    opened_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    close_reason: str | None = None

    def update_price(self, new_price: float) -> None:
        """Mark-to-market valuation."""
        self.current_price = new_price
        self.updated_at = datetime.now(UTC)
        if self.side == OrderSide.BUY:
            self.unrealized_pnl = round((new_price - self.entry_price) * self.quantity, 2)
        else:
            self.unrealized_pnl = round((self.entry_price - new_price) * self.quantity, 2)
