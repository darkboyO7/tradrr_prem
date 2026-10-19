"""Trading Intelligence Platform - Data Normalization Engine.

Converts diverse, exchange-specific payloads into canonical NormalizedTradeEvent
and NormalizedQuoteEvent models with strict UTC timestamps and decimal precisions.
"""

from datetime import UTC, datetime

from core.events import (
    NormalizedQuoteEvent,
    NormalizedTradeEvent,
    RawMarketEvent,
    TradeSide,
)
from core.logging import get_logger

logger = get_logger(__name__)


class EventNormalizer:
    """Canonical event normalizer supporting multi-venue payload transformations."""

    def normalize(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Dispatch normalization based on provider and channel type."""
        provider = raw_event.provider.lower()
        if "coinbase" in provider:
            return self._normalize_coinbase(raw_event)
        elif "binance" in provider:
            return self._normalize_binance(raw_event)
        elif "kraken" in provider:
            return self._normalize_kraken(raw_event)
        elif "nse" in provider:
            return self._normalize_nse_bse(raw_event)
        elif "bse" in provider:
            return self._normalize_nse_bse(raw_event)
        elif "forex" in provider:
            return self._normalize_forex(raw_event)
        elif "quotex" in provider:
            return self._normalize_quotex(raw_event)
        return None

    def _normalize_kraken(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Normalize Kraken trade and spread payloads."""
        payload = raw_event.raw_payload
        channel = raw_event.channel

        if channel == "trade":
            try:
                trade_time = float(payload.get("time", raw_event.received_at.timestamp()))
                exch_dt = datetime.fromtimestamp(trade_time, tz=UTC)
                latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)

                side_raw = str(payload.get("side", "")).lower()
                side = TradeSide.BUY if side_raw == "b" else (TradeSide.SELL if side_raw == "s" else TradeSide.UNKNOWN)

                return NormalizedTradeEvent(
                    provider=raw_event.provider,
                    symbol=raw_event.symbol,
                    trade_id=f"krk_{int(trade_time * 1000)}",
                    timestamp=exch_dt,
                    received_at=raw_event.received_at,
                    price=float(payload.get("price", 0.0)),
                    size=float(payload.get("volume", 0.0)),
                    side=side,
                    latency_ms=round(latency_ms, 2),
                )
            except Exception as exc:
                logger.error("Failed to normalize Kraken trade event", error=str(exc))
                return None

        elif channel == "spread":
            try:
                spread_time = float(payload.get("time", raw_event.received_at.timestamp()))
                exch_dt = datetime.fromtimestamp(spread_time, tz=UTC)
                return NormalizedQuoteEvent(
                    provider=raw_event.provider,
                    symbol=raw_event.symbol,
                    timestamp=exch_dt,
                    received_at=raw_event.received_at,
                    bid_price=float(payload.get("bid", 0.0)),
                    bid_size=float(payload.get("bid_volume", 0.0)),
                    ask_price=float(payload.get("ask", 0.0)),
                    ask_size=float(payload.get("ask_volume", 0.0)),
                    latency_ms=0.0,
                )
            except Exception as exc:
                logger.error("Failed to normalize Kraken spread event", error=str(exc))
                return None

        return None

    def _normalize_binance(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Normalize Binance trade and bookTicker payloads."""
        payload = raw_event.raw_payload
        event_type = payload.get("e", "")

        # 1. Binance Trade
        if event_type == "trade":
            try:
                trade_time_ms = payload.get("T", payload.get("E", int(raw_event.received_at.timestamp() * 1000)))
                exch_dt = datetime.fromtimestamp(trade_time_ms / 1000.0, tz=UTC)
                latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)

                # Buyer is maker (m == True) means trade was executed as a SELL taker order
                is_buyer_maker = payload.get("m", False)
                side = TradeSide.SELL if is_buyer_maker else TradeSide.BUY

                return NormalizedTradeEvent(
                    provider=raw_event.provider,
                    symbol=raw_event.symbol,
                    trade_id=str(payload.get("t", "")),
                    timestamp=exch_dt,
                    received_at=raw_event.received_at,
                    price=float(payload.get("p", 0.0)),
                    size=float(payload.get("q", 0.0)),
                    side=side,
                    sequence=payload.get("t"),
                    latency_ms=round(latency_ms, 2),
                )
            except Exception as exc:
                logger.error("Failed to normalize Binance trade event", error=str(exc), payload=payload)
                return None

        # 2. Binance BookTicker (Best Bid/Ask)
        elif "b" in payload and "a" in payload:
            try:
                return NormalizedQuoteEvent(
                    provider=raw_event.provider,
                    symbol=raw_event.symbol,
                    timestamp=raw_event.received_at,
                    received_at=raw_event.received_at,
                    bid_price=float(payload.get("b", 0.0)),
                    bid_size=float(payload.get("B", 0.0)),
                    ask_price=float(payload.get("a", 0.0)),
                    ask_size=float(payload.get("A", 0.0)),
                    sequence=payload.get("u"),
                    latency_ms=0.0,
                )
            except Exception as exc:
                logger.error("Failed to normalize Binance bookTicker event", error=str(exc))
                return None

        return None

    def _normalize_coinbase(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Normalize Coinbase Exchange WebSocket match and ticker payloads."""
        payload = raw_event.raw_payload
        msg_type = payload.get("type", "")

        # 1. Matches (Trades)
        if msg_type in ("match", "last_match"):
            try:
                time_str = payload.get("time")
                if time_str:
                    time_str = time_str.replace("Z", "+00:00")
                    exch_dt = datetime.fromisoformat(time_str)
                else:
                    exch_dt = raw_event.received_at

                latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)

                side_raw = payload.get("side", "").upper()
                side = TradeSide.BUY if side_raw == "BUY" else (TradeSide.SELL if side_raw == "SELL" else TradeSide.UNKNOWN)

                trade = NormalizedTradeEvent(
                    provider=raw_event.provider,
                    symbol=raw_event.symbol,
                    trade_id=str(payload.get("trade_id", "")),
                    timestamp=exch_dt,
                    received_at=raw_event.received_at,
                    price=float(payload.get("price", 0.0)),
                    size=float(payload.get("size", 0.0)),
                    side=side,
                    sequence=payload.get("sequence"),
                    latency_ms=round(latency_ms, 2),
                )
                return trade
            except Exception as exc:
                logger.error("Failed to normalize Coinbase match event", error=str(exc), payload=payload)
                return None

        # 2. Ticker (Quotes / Top of Book)
        elif msg_type == "ticker":
            try:
                time_str = payload.get("time")
                if time_str:
                    time_str = time_str.replace("Z", "+00:00")
                    exch_dt = datetime.fromisoformat(time_str)
                else:
                    exch_dt = raw_event.received_at

                latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)

                # If best_bid and best_ask are present
                if "best_bid" in payload and "best_ask" in payload:
                    return NormalizedQuoteEvent(
                        provider=raw_event.provider,
                        symbol=raw_event.symbol,
                        timestamp=exch_dt,
                        received_at=raw_event.received_at,
                        bid_price=float(payload.get("best_bid", 0.0)),
                        bid_size=float(payload.get("best_bid_size", 0.0)),
                        ask_price=float(payload.get("best_ask", 0.0)),
                        ask_size=float(payload.get("best_ask_size", 0.0)),
                        sequence=payload.get("sequence"),
                        latency_ms=round(latency_ms, 2),
                    )
            except Exception as exc:
                logger.error("Failed to normalize Coinbase ticker event", error=str(exc))
                return None

        return None

    def _normalize_nse_bse(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Normalize NSE and BSE equity ticks and quotes."""
        payload = raw_event.raw_payload
        channel = raw_event.channel
        symbol = raw_event.symbol

        try:
            ts = float(payload.get("timestamp", raw_event.received_at.timestamp()))
            exch_dt = datetime.fromtimestamp(ts, tz=UTC)
            latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)

            if channel == "trade" or ("price" in payload and "bid" not in payload):
                side_raw = str(payload.get("side", "")).lower()
                side = TradeSide.BUY if side_raw == "buy" else (TradeSide.SELL if side_raw == "sell" else TradeSide.UNKNOWN)
                return NormalizedTradeEvent(
                    provider=raw_event.provider,
                    symbol=symbol,
                    trade_id=str(payload.get("trade_id", f"in_{int(ts * 1000)}")),
                    timestamp=exch_dt,
                    received_at=raw_event.received_at,
                    price=float(payload.get("price", 0.0)),
                    size=float(payload.get("volume", payload.get("quantity", 1.0))),
                    side=side,
                    latency_ms=round(latency_ms, 2),
                )
            elif channel in ("quote", "depth") or "bid" in payload:
                return NormalizedQuoteEvent(
                    provider=raw_event.provider,
                    symbol=symbol,
                    timestamp=exch_dt,
                    received_at=raw_event.received_at,
                    bid_price=float(payload.get("bid", payload.get("price", 0.0) - 0.05)),
                    bid_size=float(payload.get("bid_qty", 100.0)),
                    ask_price=float(payload.get("ask", payload.get("price", 0.0) + 0.05)),
                    ask_size=float(payload.get("ask_qty", 100.0)),
                    latency_ms=round(latency_ms, 2),
                )
        except Exception as exc:
            logger.error("Failed to normalize Indian equity event", error=str(exc))
        return None

    def _normalize_forex(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Normalize Forex spreads and quotes (EUR/CHF, majors)."""
        payload = raw_event.raw_payload
        symbol = raw_event.symbol

        try:
            ts = float(payload.get("timestamp", raw_event.received_at.timestamp()))
            exch_dt = datetime.fromtimestamp(ts, tz=UTC)
            latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)

            bid = float(payload.get("bid", payload.get("price", 0.0)))
            ask = float(payload.get("ask", bid + float(payload.get("spread", 0.00015))))

            return NormalizedQuoteEvent(
                provider=raw_event.provider,
                symbol=symbol,
                timestamp=exch_dt,
                received_at=raw_event.received_at,
                bid_price=bid,
                bid_size=float(payload.get("volume", 100000.0)),
                ask_price=ask,
                ask_size=float(payload.get("volume", 100000.0)),
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            logger.error("Failed to normalize Forex event", error=str(exc))
        return None

    def _normalize_quotex(self, raw_event: RawMarketEvent) -> NormalizedTradeEvent | NormalizedQuoteEvent | None:
        """Normalize Quotex tick events."""
        payload = raw_event.raw_payload
        symbol = raw_event.symbol

        try:
            ts = float(payload.get("timestamp", raw_event.received_at.timestamp()))
            exch_dt = datetime.fromtimestamp(ts, tz=UTC)
            latency_ms = max(0.0, (raw_event.received_at - exch_dt).total_seconds() * 1000.0)
            price = float(payload.get("price", 0.0))

            return NormalizedQuoteEvent(
                provider=raw_event.provider,
                symbol=symbol,
                timestamp=exch_dt,
                received_at=raw_event.received_at,
                bid_price=price,
                bid_size=100.0,
                ask_price=price,
                ask_size=100.0,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as exc:
            logger.error("Failed to normalize Quotex event", error=str(exc))
        return None
