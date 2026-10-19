"""Trading Intelligence Platform - Real-Time Chronological Candle Engine.

Aggregates high-frequency normalized trade ticks into canonical 1-minute (and higher)
candlesticks, computing VWAP, volume, quote volume, and trade counts with clock alignment.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from core.events import Candle, NormalizedTradeEvent
from core.logging import get_logger

logger = get_logger(__name__)


def floor_to_minute(dt: datetime) -> datetime:
    """Floor a datetime to the start of its 1-minute interval."""
    return dt.replace(second=0, microsecond=0)


class ActiveCandleState:
    """Mutable accumulator for an in-progress candle bar."""

    def __init__(self, symbol: str, timeframe: str, start_time: datetime, initial_price: float, initial_size: float) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.timestamp = start_time
        self.open = initial_price
        self.high = initial_price
        self.low = initial_price
        self.close = initial_price
        self.volume = initial_size
        self.quote_volume = initial_price * initial_size
        self.trade_count = 1

    def update(self, price: float, size: float) -> None:
        """Update bar metrics with a new trade tick."""
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += size
        self.quote_volume += price * size
        self.trade_count += 1

    def to_immutable(self, is_closed: bool) -> Candle:
        """Export as an immutable canonical Candle instance."""
        vwap = (self.quote_volume / self.volume) if self.volume > 0 else self.close
        return Candle(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=round(self.volume, 8),
            quote_volume=round(self.quote_volume, 4),
            trade_count=self.trade_count,
            vwap=round(vwap, 4),
            is_closed=is_closed,
        )


class CandleEngine:
    """Stateful chronological 1-minute candle builder."""

    def __init__(
        self,
        timeframe: str = "1m",
        on_candle_closed: Callable[[Candle], Any] | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.on_candle_closed = on_candle_closed
        self._active_bars: dict[str, ActiveCandleState] = {}
        self._last_closed_bars: dict[str, Candle] = {}

    def process_trade(self, trade: NormalizedTradeEvent) -> list[Candle]:
        """Ingest a normalized trade tick, updating the active bar or closing the previous bar."""
        closed_candles: list[Candle] = []
        bar_start = floor_to_minute(trade.timestamp)
        symbol = trade.symbol

        active_bar = self._active_bars.get(symbol)

        if active_bar is None:
            # First bar for this symbol
            self._active_bars[symbol] = ActiveCandleState(
                symbol=symbol,
                timeframe=self.timeframe,
                start_time=bar_start,
                initial_price=trade.price,
                initial_size=trade.size,
            )
        elif bar_start == active_bar.timestamp:
            # Tick falls in the same active 1-minute window
            active_bar.update(trade.price, trade.size)
        elif bar_start > active_bar.timestamp:
            # New minute has started! Finalize previous bar
            finalized = active_bar.to_immutable(is_closed=True)
            closed_candles.append(finalized)
            self._last_closed_bars[symbol] = finalized

            if self.on_candle_closed:
                try:
                    self.on_candle_closed(finalized)
                except Exception as exc:
                    logger.error("Error executing on_candle_closed callback", error=str(exc))

            # Initialize new active bar
            self._active_bars[symbol] = ActiveCandleState(
                symbol=symbol,
                timeframe=self.timeframe,
                start_time=bar_start,
                initial_price=trade.price,
                initial_size=trade.size,
            )
        else:
            # Late-arriving historical tick from past window
            logger.warning(
                "Late arriving tick ignored for active candle aggregation",
                symbol=symbol,
                tick_time=trade.timestamp.isoformat(),
                bar_time=active_bar.timestamp.isoformat()
            )

        return closed_candles

    def get_current_partial_candle(self, symbol: str) -> Candle | None:
        """Retrieve current in-flight partial candle bar."""
        active = self._active_bars.get(symbol)
        if active:
            return active.to_immutable(is_closed=False)
        return None

    def get_last_closed_candle(self, symbol: str) -> Candle | None:
        """Retrieve most recently finalized candle bar."""
        return self._last_closed_bars.get(symbol)

    def force_close_all(self) -> list[Candle]:
        """Force finalize all currently active bars (e.g. on shutdown)."""
        closed: list[Candle] = []
        for symbol, active in list(self._active_bars.items()):
            finalized = active.to_immutable(is_closed=True)
            closed.append(finalized)
            self._last_closed_bars[symbol] = finalized
            if self.on_candle_closed:
                try:
                    self.on_candle_closed(finalized)
                except Exception as exc:
                    logger.error("Error executing on_candle_closed callback", error=str(exc))
        self._active_bars.clear()
        return closed
