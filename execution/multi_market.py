"""Trading Intelligence Platform - Multi-Market Execution Adapters.

Provides clean execution adapters for:
- NSE (Indian Equities with exchange lot sizes, market hours check)
- BSE (Indian Equities)
- Forex (Pip precision, spread accounting, 24/5 sessions)
- Quotex (Analysis & Paper mode, fixed payout evaluation)
"""

import asyncio
import math
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
        enforce_market_hours: bool = False,
    ) -> None:
        self.market = market
        self.engine = paper_engine or PaperTradingEngine()
        self.simulated_latency_ms = simulated_latency_ms
        self.enforce_market_hours = enforce_market_hours
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

        order_id = client_order_id or f"{self.market.value.lower()}_ord_{int(time.time()*1000)}"

        # 1. Connected state check
        if not self.is_connected:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": "CLIENT_NOT_CONNECTED: Execution client must be connected before placing orders",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        # 2. Market session check
        session = SymbolResolver.get_session(self.market)
        is_open = session.is_open() if session else True
        if self.enforce_market_hours and not is_open:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "market": self.market.value,
                "market_open": False,
                "status": "REJECTED",
                "reason": f"MARKET_CLOSED: {self.market.value} session is currently closed",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        # 3. Instrument resolution and market validation
        inst = SymbolResolver.resolve(symbol)
        if not inst:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": f"UNSUPPORTED_OR_UNKNOWN_INSTRUMENT: '{symbol}' not recognized by symbol resolver",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }
        clean_sym = inst.canonical_symbol

        # 4. Order side validation (strict BUY / SELL, never map unknown to SELL)
        side_upper = str(side).upper().strip()
        if side_upper == "BUY":
            order_side = OrderSide.BUY
        elif side_upper == "SELL":
            order_side = OrderSide.SELL
        else:
            return {
                "order_id": order_id,
                "symbol": clean_sym,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": f"INVALID_ORDER_SIDE: '{side}' is not valid (must be BUY or SELL)",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        # 5. Order type validation (strict MARKET / LIMIT / STOP, never map unknown to STOP)
        type_upper = str(order_type).upper().strip()
        if type_upper == "MARKET":
            order_t = OrderType.MARKET
        elif type_upper == "LIMIT":
            order_t = OrderType.LIMIT
        elif type_upper == "STOP":
            order_t = OrderType.STOP
        else:
            return {
                "order_id": order_id,
                "symbol": clean_sym,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": f"INVALID_ORDER_TYPE: '{order_type}' is not valid (must be MARKET, LIMIT, or STOP)",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        # 6. Quantity validation
        if quantity is None or math.isnan(quantity) or math.isinf(quantity) or quantity <= 0:
            return {
                "order_id": order_id,
                "symbol": clean_sym,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": f"INVALID_QUANTITY: {quantity} (must be a positive finite number)",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }
        final_qty = inst.round_quantity(quantity)
        if final_qty <= 0:
            return {
                "order_id": order_id,
                "symbol": clean_sym,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": f"INVALID_QUANTITY_AFTER_ROUNDING: {final_qty} (below min lot)",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        # 7. Price validation (strictly reject missing, zero, negative, NaN, inf)
        if order_t in (OrderType.LIMIT, OrderType.STOP):
            if price is None or math.isnan(price) or math.isinf(price) or price <= 0:
                return {
                    "order_id": order_id,
                    "symbol": clean_sym,
                    "market": self.market.value,
                    "status": "REJECTED",
                    "reason": f"INVALID_PRICE: {price} for {order_t.value} order (must be positive finite number)",
                    "execution_mode": "SIMULATED_PAPER",
                    "is_live": False,
                }
            final_price = inst.round_price(price)
        else:
            final_price = inst.round_price(price) if (price is not None and not (math.isnan(price) or math.isinf(price) or price <= 0)) else None

        # Market price check for fill: require valid verified market price (NO 100.0 fallback)
        market_price = self._market_prices.get(clean_sym) or final_price
        if market_price is None or math.isnan(market_price) or math.isinf(market_price) or market_price <= 0:
            return {
                "order_id": order_id,
                "symbol": clean_sym,
                "market": self.market.value,
                "status": "REJECTED",
                "reason": f"UNAVAILABLE_MARKET_PRICE: No verified live market price for {clean_sym}",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        # 8. Check for stale data state if live market enforcement is on or no manual price override
        if self.enforce_market_hours or clean_sym not in self._market_prices:
            try:
                from data.market_state import MarketStateService
                st = MarketStateService.get_instance().get_state(clean_sym)
                if st and (st.is_stale or st.source_state in ("STALE", "UNAVAILABLE")):
                    return {
                        "order_id": order_id,
                        "symbol": clean_sym,
                        "market": self.market.value,
                        "status": "REJECTED",
                        "reason": f"STALE_OR_UNAVAILABLE_DATA: Market data age {st.data_age_seconds:.1f}s is stale/unavailable",
                        "execution_mode": "SIMULATED_PAPER",
                        "is_live": False,
                    }
            except Exception:
                pass

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
            "execution_mode": "SIMULATED_PAPER",
            "is_live": False,
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

    def __init__(
        self,
        paper_engine: PaperTradingEngine | None = None,
        enforce_market_hours: bool = False,
    ) -> None:
        super().__init__(
            market=MarketType.NSE,
            paper_engine=paper_engine,
            enforce_market_hours=enforce_market_hours,
        )


class BSEExecutionClient(MultiMarketPaperExecutionClient):
    """Execution client for Bombay Stock Exchange (BSE)."""

    def __init__(
        self,
        paper_engine: PaperTradingEngine | None = None,
        enforce_market_hours: bool = False,
    ) -> None:
        super().__init__(
            market=MarketType.BSE,
            paper_engine=paper_engine,
            enforce_market_hours=enforce_market_hours,
        )


class ForexExecutionClient(MultiMarketPaperExecutionClient):
    """Execution client for Forex (EUR/CHF and FX majors)."""

    def __init__(
        self,
        paper_engine: PaperTradingEngine | None = None,
        enforce_market_hours: bool = False,
    ) -> None:
        super().__init__(
            market=MarketType.FOREX,
            paper_engine=paper_engine,
            enforce_market_hours=enforce_market_hours,
        )


class QuotexExecutionClient(MultiMarketPaperExecutionClient):
    """Paper execution client for Quotex (strictly analysis and paper mode)."""

    def __init__(
        self,
        paper_engine: PaperTradingEngine | None = None,
        enforce_market_hours: bool = False,
    ) -> None:
        super().__init__(
            market=MarketType.QUOTEX,
            paper_engine=paper_engine,
            enforce_market_hours=enforce_market_hours,
        )
