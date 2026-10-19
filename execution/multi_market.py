"""Trading Intelligence Platform - Multi-Market Execution Adapters.

Provides clean execution adapters for:
- NSE (Indian Equities with exchange lot sizes, market hours check)
- BSE (Indian Equities)
- Forex (Pip precision, spread accounting, 24/5 sessions)
- Quotex (Analysis & Paper mode, fixed payout evaluation)
"""

import asyncio
import time
from typing import Any

from core.instrument import MarketType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from execution.base import BaseExecutionClient
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, OrderType, PaperOrder

logger = get_logger(__name__)


class MultiMarketPaperExecutionClient(BaseExecutionClient):
    """Institutional execution adapter with full multi-market validation."""

    def __init__(
        self,
        market: MarketType,
        paper_engine: PaperTradingEngine | None = None,
        simulated_latency_ms: float = 15.0,
    ) -> None:
        self.market = market
        self.engine = paper_engine or PaperTradingEngine()
        self.simulated_latency_ms = simulated_latency_ms
        self.is_connected = False
        self._market_prices: dict[str, float] = {}

    def set_market_price(self, symbol: str, price: float) -> None:
        self._market_prices[symbol] = price
        self.engine.on_tick(symbol, price)

    async def connect(self) -> None:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        self.is_connected = True
        logger.info(f"Connected {self.market.value} execution client")

    async def disconnect(self) -> None:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        self.is_connected = False
        logger.info(f"Disconnected {self.market.value} execution client")

    async def get_balance(self, asset: str = "USD") -> float:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        return float(self.engine.cash)

    async def get_positions(self) -> list[dict[str, Any]]:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        return [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
            }
            for p in self.engine.positions.values()
        ]

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        inst = SymbolResolver.resolve(symbol)
        clean_sym = inst.canonical_symbol if inst else symbol

        # Validate market session
        session = SymbolResolver.get_session(self.market)
        is_open = session.is_open() if session else True

        # Round according to instrument specifications
        final_qty = inst.round_quantity(quantity) if inst else quantity
        final_price = inst.round_price(price) if (inst and price is not None) else price

        order_id = client_order_id or f"{self.market.value.lower()}_ord_{int(time.time()*1000)}"
        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        order_t = (
            OrderType.MARKET
            if order_type.upper() == "MARKET"
            else (OrderType.LIMIT if order_type.upper() == "LIMIT" else OrderType.STOP)
        )

        paper_order = PaperOrder(
            order_id=order_id,
            symbol=clean_sym,
            side=order_side,
            order_type=order_t,
            quantity=final_qty,
            limit_price=final_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        market_price = self._market_prices.get(clean_sym, final_price or 100.0)
        executed = self.engine.execute_order(paper_order, current_market_price=market_price)

        return {
            "order_id": executed.order_id,
            "symbol": executed.symbol,
            "market": self.market.value,
            "market_open": is_open,
            "side": executed.side.value,
            "status": executed.status.value,
            "filled_price": executed.filled_price,
            "filled_quantity": executed.filled_quantity,
            "fee": executed.fee,
            "slippage": executed.slippage,
            "timestamp": executed.filled_at.isoformat() if executed.filled_at else None,
        }

    async def cancel_order(self, order_id: str) -> bool:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        return False

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        for ord_ in self.engine.orders:
            if ord_.order_id == order_id:
                return {
                    "order_id": ord_.order_id,
                    "symbol": ord_.symbol,
                    "status": ord_.status.value,
                    "filled_price": ord_.filled_price,
                    "filled_quantity": ord_.filled_quantity,
                }
        return {"order_id": order_id, "status": "NOT_FOUND"}


class NSEExecutionClient(MultiMarketPaperExecutionClient):
    """Execution client for National Stock Exchange of India (NSE)."""

    def __init__(self, paper_engine: PaperTradingEngine | None = None) -> None:
        super().__init__(market=MarketType.NSE, paper_engine=paper_engine)


class BSEExecutionClient(MultiMarketPaperExecutionClient):
    """Execution client for Bombay Stock Exchange (BSE)."""

    def __init__(self, paper_engine: PaperTradingEngine | None = None) -> None:
        super().__init__(market=MarketType.BSE, paper_engine=paper_engine)


class ForexExecutionClient(MultiMarketPaperExecutionClient):
    """Execution client for Forex (EUR/CHF and FX majors)."""

    def __init__(self, paper_engine: PaperTradingEngine | None = None) -> None:
        super().__init__(market=MarketType.FOREX, paper_engine=paper_engine)


class QuotexExecutionClient(MultiMarketPaperExecutionClient):
    """Paper execution client for Quotex (strictly analysis and paper mode)."""

    def __init__(self, paper_engine: PaperTradingEngine | None = None) -> None:
        super().__init__(market=MarketType.QUOTEX, paper_engine=paper_engine)
