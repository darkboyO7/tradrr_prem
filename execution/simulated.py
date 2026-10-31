"""Trading Intelligence Platform - Simulated Execution Client.

Provides an in-memory execution adapter wrapping the PaperTradingEngine
to implement the full BaseExecutionClient protocol with simulated network latency.
"""

import asyncio
import math
import time
from typing import Any

from execution.base import BaseExecutionClient
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, OrderType, PaperOrder


class SimulatedExecutionClient(BaseExecutionClient):
    """High-fidelity virtual broker client implementing BaseExecutionClient."""

    def __init__(
        self,
        paper_engine: PaperTradingEngine | None = None,
        simulated_latency_ms: float = 10.0,
    ) -> None:
        self.engine = paper_engine or PaperTradingEngine()
        self.simulated_latency_ms = simulated_latency_ms
        self.is_connected = False
        self._market_prices: dict[str, float] = {}

    def set_market_price(self, symbol: str, price: float) -> None:
        """Update simulated reference market price."""
        self._market_prices[symbol] = price
        self.engine.on_tick(symbol, price)

    async def connect(self) -> None:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        self.is_connected = True

    async def disconnect(self) -> None:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        self.is_connected = False

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

        order_id = client_order_id or f"sim_ord_{int(time.time()*1000)}"

        if not self.is_connected:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "REJECTED",
                "reason": "CLIENT_NOT_CONNECTED",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        side_upper = str(side).upper().strip()
        if side_upper == "BUY":
            order_side = OrderSide.BUY
        elif side_upper == "SELL":
            order_side = OrderSide.SELL
        else:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "REJECTED",
                "reason": f"INVALID_ORDER_SIDE: {side}",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

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
                "symbol": symbol,
                "status": "REJECTED",
                "reason": f"INVALID_ORDER_TYPE: {order_type}",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        if quantity is None or math.isnan(quantity) or math.isinf(quantity) or quantity <= 0:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "REJECTED",
                "reason": f"INVALID_QUANTITY: {quantity}",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        if order_t in (OrderType.LIMIT, OrderType.STOP):
            if price is None or math.isnan(price) or math.isinf(price) or price <= 0:
                return {
                    "order_id": order_id,
                    "symbol": symbol,
                    "status": "REJECTED",
                    "reason": f"INVALID_PRICE: {price}",
                    "execution_mode": "SIMULATED_PAPER",
                    "is_live": False,
                }

        market_price = self._market_prices.get(symbol) or price
        if market_price is None or math.isnan(market_price) or math.isinf(market_price) or market_price <= 0:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "REJECTED",
                "reason": f"UNAVAILABLE_MARKET_PRICE: No verified price for {symbol}",
                "execution_mode": "SIMULATED_PAPER",
                "is_live": False,
            }

        paper_order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=order_side,
            order_type=order_t,
            quantity=quantity,
            limit_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        executed = self.engine.execute_order(paper_order, current_market_price=market_price)

        return {
            "order_id": executed.order_id,
            "symbol": executed.symbol,
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
        # Market orders in simulation fill immediately
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
