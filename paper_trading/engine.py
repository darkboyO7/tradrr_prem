"""Trading Intelligence Platform - Paper Trading Engine.

Simulates an institutional brokerage account with real-time portfolio valuation,
active position management, and automatic stop-loss/take-profit order execution.
"""

import math
from datetime import UTC, datetime
from typing import Any

from core.database import get_db_session
from core.events import Candle
from core.logging import get_logger
from data.storage.models import PaperTradeModel
from paper_trading.models import OrderSide, OrderStatus, PaperOrder, PaperPosition
from paper_trading.simulator import ExecutionSimulator

logger = get_logger(__name__)


class PaperTradingEngine:
    """Manages virtual brokerage state, order routing, fills, and stop-loss monitoring."""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        simulator: ExecutionSimulator | None = None,
        session_factory: Any | None = None,
    ) -> None:
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.simulator = simulator or ExecutionSimulator()
        self._session_factory = session_factory or get_db_session

        self.positions: dict[str, PaperPosition] = {}  # symbol -> active PaperPosition
        self.closed_positions: list[PaperPosition] = []
        self.orders: list[PaperOrder] = []

    @property
    def equity(self) -> float:
        """Total portfolio equity (cash balance + sum of unrealized PnL)."""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return round(self.cash + unrealized, 2)

    def execute_order(self, order: PaperOrder, current_market_price: float) -> PaperOrder:
        """Process and fill an incoming virtual order."""
        if (
            current_market_price is None
            or math.isnan(current_market_price)
            or math.isinf(current_market_price)
            or current_market_price <= 0
        ):
            order.status = OrderStatus.REJECTED
            logger.warning("Paper order rejected: invalid market price", order_id=order.order_id, price=current_market_price)
            self.orders.append(order)
            return order

        if (
            order.quantity is None
            or math.isnan(order.quantity)
            or math.isinf(order.quantity)
            or order.quantity <= 0
        ):
            order.status = OrderStatus.REJECTED
            logger.warning("Paper order rejected: invalid quantity", order_id=order.order_id, qty=order.quantity)
            self.orders.append(order)
            return order

        self.orders.append(order)

        fill_price, fee, slippage = self.simulator.simulate_fill(order, current_market_price)
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.fee = fee
        order.slippage = slippage
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now(UTC)

        # Deduct transaction fee from cash
        self.cash -= fee

        symbol = order.symbol

        # Check if closing an existing position or opening a new one
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Opposite side closes or reduces position
            if (pos.side == OrderSide.BUY and order.side == OrderSide.SELL) or (
                pos.side == OrderSide.SELL and order.side == OrderSide.BUY
            ):
                self._close_position(pos, fill_price, reason="ORDER_EXIT")
            else:
                # Adding to existing position: weighted average entry price
                total_qty = pos.quantity + order.quantity
                avg_entry = ((pos.entry_price * pos.quantity) + (fill_price * order.quantity)) / total_qty
                pos.quantity = total_qty
                pos.entry_price = round(avg_entry, 4)
                pos.update_price(current_market_price)
        else:
            # Open new position
            sl = order.stop_loss if order.stop_loss is not None else (fill_price * 0.98 if order.side == OrderSide.BUY else fill_price * 1.02)
            tp = order.take_profit if order.take_profit is not None else (fill_price * 1.04 if order.side == OrderSide.BUY else fill_price * 0.96)

            new_pos = PaperPosition(
                position_id=f"pos_{order.order_id}",
                symbol=symbol,
                side=order.side,
                quantity=order.quantity,
                entry_price=fill_price,
                current_price=fill_price,
                stop_loss=sl,
                take_profit=tp,
            )
            self.positions[symbol] = new_pos

        logger.info(
            "Paper order executed",
            symbol=symbol,
            side=order.side.value,
            qty=order.quantity,
            fill_price=fill_price,
            fee=fee,
            slippage=slippage,
        )
        return order

    def _close_position(self, pos: PaperPosition, exit_price: float, reason: str) -> None:
        """Finalize an active position, realize PnL, and credit cash."""
        pos.update_price(exit_price)
        pos.is_open = False
        pos.closed_at = datetime.now(UTC)
        pos.close_reason = reason

        if pos.side == OrderSide.BUY:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pos.realized_pnl = round(pnl, 2)
        pos.unrealized_pnl = 0.0

        # Credit cash balance with realized PnL
        self.cash += pos.realized_pnl
        self.closed_positions.append(pos)
        self.positions.pop(pos.symbol, None)

        logger.info(
            "Position closed",
            symbol=pos.symbol,
            side=pos.side.value,
            entry=pos.entry_price,
            exit=exit_price,
            realized_pnl=pos.realized_pnl,
            reason=reason,
        )

    def on_tick(self, symbol: str, price: float) -> list[str]:
        """Update market price, MTM valuation, and trigger stop-loss or take-profit auto-exits."""
        events: list[str] = []
        if symbol not in self.positions:
            return events

        pos = self.positions[symbol]
        pos.update_price(price)

        # Check Stop Loss & Take Profit
        if pos.side == OrderSide.BUY:
            if price <= pos.stop_loss:
                reason = f"STOP_LOSS_HIT (Price: {price} <= SL: {pos.stop_loss})"
                self._close_position(pos, price, reason=reason)
                events.append(reason)
            elif price >= pos.take_profit:
                reason = f"TAKE_PROFIT_HIT (Price: {price} >= TP: {pos.take_profit})"
                self._close_position(pos, price, reason=reason)
                events.append(reason)
        elif pos.side == OrderSide.SELL:
            if price >= pos.stop_loss:
                reason = f"STOP_LOSS_HIT (Price: {price} >= SL: {pos.stop_loss})"
                self._close_position(pos, price, reason=reason)
                events.append(reason)
            elif price <= pos.take_profit:
                reason = f"TAKE_PROFIT_HIT (Price: {price} <= TP: {pos.take_profit})"
                self._close_position(pos, price, reason=reason)
                events.append(reason)

        return events

    def on_candle(self, candle: Candle) -> list[str]:
        """Process finalized candle bar for position valuation and bounds checks."""
        events = []
        symbol = candle.symbol
        if symbol in self.positions:
            # Check high and low breaches first
            pos = self.positions[symbol]
            if pos.side == OrderSide.BUY:
                if candle.low <= pos.stop_loss:
                    events.extend(self.on_tick(symbol, pos.stop_loss))
                elif candle.high >= pos.take_profit:
                    events.extend(self.on_tick(symbol, pos.take_profit))
                else:
                    events.extend(self.on_tick(symbol, candle.close))
            elif pos.side == OrderSide.SELL:
                if candle.high >= pos.stop_loss:
                    events.extend(self.on_tick(symbol, pos.stop_loss))
                elif candle.low <= pos.take_profit:
                    events.extend(self.on_tick(symbol, pos.take_profit))
                else:
                    events.extend(self.on_tick(symbol, candle.close))
        return events

    async def persist_trade_to_db(self, order: PaperOrder) -> int | None:
        """Save paper execution to SQL database."""
        try:
            async with self._session_factory() as session:
                record = PaperTradeModel(
                    trade_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.order_type.value,
                    requested_price=order.limit_price or order.filled_price or 0.0,
                    filled_price=order.filled_price,
                    quantity=order.quantity,
                    slippage=order.slippage,
                    commission=order.fee,
                    status=order.status.value,
                    submitted_at=order.submitted_at,
                    filled_at=order.filled_at,
                )
                session.add(record)
                await session.commit()
                return record.id
        except Exception as exc:
            logger.error("Failed to persist paper trade to DB", error=str(exc))
            return None
