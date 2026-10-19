"""Trading Intelligence Platform - Multi-Timeframe Streaming Aggregator.

Maintains in-memory rolling state for multiple concurrent timeframes (e.g. 1m, 5m, 15m, 1h, 4h, 1d)
and produces closed higher-timeframe candles incrementally as 1-minute bars finalize.
"""

from collections.abc import Callable
from typing import Any

from core.events import Candle
from core.logging import get_logger
from data.candle.engine import ActiveCandleState
from data.candle.resampler import floor_time_to_timeframe, parse_timeframe_to_minutes

logger = get_logger(__name__)


class MultiTimeframeAggregator:
    """Stateful aggregator managing simultaneous candlestick timeframes for trading symbols."""

    def __init__(
        self,
        timeframes: list[str] | None = None,
        on_candle_closed: Callable[[Candle], Any] | None = None,
    ) -> None:
        # Default standard multi-timeframe hierarchy
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.on_candle_closed = on_candle_closed

        # Structure: _active_bars[symbol][timeframe] -> ActiveCandleState
        self._active_bars: dict[str, dict[str, ActiveCandleState]] = {}
        # Structure: _latest_closed[symbol][timeframe] -> Candle
        self._latest_closed: dict[str, dict[str, Candle]] = {}

    def process_candle(self, candle: Candle) -> list[Candle]:
        """Ingest a completed base candle (typically 1m) and update higher-timeframe bars.

        Returns a list of any higher-timeframe candles that closed as a result of this candle.
        """
        closed_bars: list[Candle] = []
        symbol = candle.symbol

        if symbol not in self._active_bars:
            self._active_bars[symbol] = {}
            self._latest_closed[symbol] = {}

        # Always record base 1m bar
        self._latest_closed[symbol][candle.timeframe] = candle

        # Process each higher timeframe
        for tf in self.timeframes:
            if tf == candle.timeframe:
                continue

            tf_minutes = parse_timeframe_to_minutes(tf)
            base_minutes = parse_timeframe_to_minutes(candle.timeframe)
            if tf_minutes <= base_minutes:
                continue

            bar_boundary = floor_time_to_timeframe(candle.timestamp, tf)
            active = self._active_bars[symbol].get(tf)

            if active is None:
                # Initialize new higher timeframe bar
                active = ActiveCandleState(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=bar_boundary,
                    initial_price=candle.open,
                    initial_size=candle.volume,
                )
                active.high = candle.high
                active.low = candle.low
                active.close = candle.close
                active.quote_volume = candle.quote_volume
                active.trade_count = candle.trade_count
                self._active_bars[symbol][tf] = active

            elif bar_boundary == active.timestamp:
                # Still within the current higher-timeframe boundary: aggregate
                if candle.high > active.high:
                    active.high = candle.high
                if candle.low < active.low:
                    active.low = candle.low
                active.close = candle.close
                active.volume += candle.volume
                active.quote_volume += candle.quote_volume
                active.trade_count += candle.trade_count

            elif bar_boundary > active.timestamp:
                # Timeframe boundary crossed! Finalize previous higher-timeframe candle
                finalized = active.to_immutable(is_closed=True)
                closed_bars.append(finalized)
                self._latest_closed[symbol][tf] = finalized

                if self.on_candle_closed:
                    try:
                        self.on_candle_closed(finalized)
                    except Exception as exc:
                        logger.error("Error in multi-timeframe on_candle_closed callback", error=str(exc))

                # Initialize next higher-timeframe candle
                new_active = ActiveCandleState(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=bar_boundary,
                    initial_price=candle.open,
                    initial_size=candle.volume,
                )
                new_active.high = candle.high
                new_active.low = candle.low
                new_active.close = candle.close
                new_active.quote_volume = candle.quote_volume
                new_active.trade_count = candle.trade_count
                self._active_bars[symbol][tf] = new_active

        return closed_bars

    def get_latest_candle(self, symbol: str, timeframe: str) -> Candle | None:
        """Get the most recently closed candle for a given symbol and timeframe."""
        return self._latest_closed.get(symbol, {}).get(timeframe)

    def get_context(self, symbol: str) -> dict[str, Candle | None]:
        """Return the latest closed candles across all configured timeframes for a symbol."""
        result = {}
        for tf in self.timeframes:
            result[tf] = self.get_latest_candle(symbol, tf)
        return result

    def force_close_all(self) -> list[Candle]:
        """Force finalize all currently open higher-timeframe bars (e.g. on shutdown)."""
        closed_bars: list[Candle] = []
        for symbol, tf_dict in self._active_bars.items():
            for tf, active in list(tf_dict.items()):
                finalized = active.to_immutable(is_closed=True)
                closed_bars.append(finalized)
                self._latest_closed[symbol][tf] = finalized
                if self.on_candle_closed:
                    try:
                        self.on_candle_closed(finalized)
                    except Exception as exc:
                        logger.error("Error in on_candle_closed on force close", error=str(exc))
            tf_dict.clear()
        return closed_bars
