"""Trading Intelligence Platform - Binance Market Data Provider.

Supports streaming public trade and kline WebSocket feeds from Binance,
normalizing Binance-specific payload structures into canonical RawMarketEvent objects.
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from core.events import Candle, ProviderState, RawMarketEvent
from core.logging import get_logger
from data.ingestion.base import MarketDataProvider

logger = get_logger("binance_ws")


class BinanceWebSocketProvider(MarketDataProvider):
    """High-performance WebSocket connector for Binance public market data."""

    BASE_WS_URL = "wss://stream.binance.com:9443/ws"
    BASE_REST_URL = "https://api.binance.com/api/v3"

    def __init__(
        self,
        symbols: list[str],
        channels: list[str] | None = None,
        stale_threshold_seconds: float = 15.0,
        ws_url: str | None = None,
        rest_url: str | None = None,
    ) -> None:
        super().__init__(name="binance", stale_threshold_seconds=stale_threshold_seconds)
        self.symbols = [s.lower().replace("-", "").replace("/", "") for s in symbols]
        self.raw_symbols = symbols
        self.channels = channels or ["trade"]
        self.ws_url = ws_url or self.BASE_WS_URL
        self.rest_url = rest_url or self.BASE_REST_URL
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._running = False
        self._stream_names: list[str] = []

    def _build_stream_names(self) -> list[str]:
        """Construct Binance combined stream parameter names."""
        streams = []
        for symbol in self.symbols:
            for channel in self.channels:
                if channel == "trade":
                    streams.append(f"{symbol}@trade")
                elif channel.startswith("kline"):
                    streams.append(f"{symbol}@{channel}")
                elif channel == "ticker":
                    streams.append(f"{symbol}@ticker")
                elif channel == "bookTicker":
                    streams.append(f"{symbol}@bookTicker")
                else:
                    streams.append(f"{symbol}@{channel}")
        return streams

    async def connect(self) -> None:
        """Establish WebSocket connection with Binance."""
        self.set_state(ProviderState.CONNECTING)
        self._stream_names = self._build_stream_names()

        # Binance combined stream URL or single stream URL
        if len(self._stream_names) == 1:
            connect_url = f"{self.ws_url}/{self._stream_names[0]}"
        else:
            streams_param = "/".join(self._stream_names)
            connect_url = f"wss://stream.binance.com:9443/stream?streams={streams_param}"

        try:
            self._ws = await websockets.connect(
                connect_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            self._running = True
            self.set_state(ProviderState.CONNECTED)
            self.start_watchdog()
            logger.info("Connected to Binance WebSocket", streams=self._stream_names)
        except Exception as e:
            self.set_state(ProviderState.ERROR if hasattr(ProviderState, "ERROR") else ProviderState.FAILED)
            logger.error("Failed to connect to Binance WebSocket", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Gracefully disconnect from Binance WebSocket."""
        self._running = False
        self.stop_watchdog()
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.set_state(ProviderState.DISCONNECTED)
        logger.info("Disconnected from Binance WebSocket")

    async def subscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Dynamically subscribe to Binance streams via JSON-RPC."""
        for s in symbols:
            self._subscribed_symbols.add(s)
        for c in channels:
            self._subscribed_channels.add(c)

        if not self._ws or not self._running:
            return

        new_streams = []
        for s in symbols:
            clean_s = s.lower().replace("-", "").replace("/", "")
            for c in channels:
                new_streams.append(f"{clean_s}@{c}")

        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": new_streams,
            "id": int(time.time() * 1000),
        }
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info("Sent Binance SUBSCRIBE message", params=new_streams)

    async def unsubscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Unsubscribe from Binance streams via JSON-RPC."""
        if not self._ws or not self._running:
            return

        streams_to_remove = []
        for s in symbols:
            clean_s = s.lower().replace("-", "").replace("/", "")
            for c in channels:
                streams_to_remove.append(f"{clean_s}@{c}")

        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": streams_to_remove,
            "id": int(time.time() * 1000),
        }
        await self._ws.send(json.dumps(unsubscribe_msg))
        logger.info("Sent Binance UNSUBSCRIBE message", params=streams_to_remove)

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        """Continuously yield incoming RawMarketEvent objects from Binance feed."""
        backoff_seconds = 1.0

        while self._running:
            try:
                if not self._ws:
                    await self.connect()

                assert self._ws is not None
                raw_msg = await self._ws.recv()
                receive_time = datetime.now(UTC)

                data = json.loads(raw_msg)
                # Unwrap combined stream wrapper if present: {"stream": "...", "data": {...}}
                payload = data.get("data", data)

                # Skip subscription acknowledgement responses
                if "result" in payload and "id" in payload:
                    continue

                event = self._parse_binance_message(payload, receive_time)
                if event:
                    self.update_freshness()
                    yield event

                backoff_seconds = 1.0

            except ConnectionClosed as e:
                logger.warning("Binance connection lost, reconnecting...", error=str(e))
                self.set_state(ProviderState.RECONNECTING)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 30.0)
                try:
                    await self.connect()
                except Exception:
                    pass
            except Exception as e:
                logger.error("Error receiving from Binance WebSocket", error=str(e))
                await asyncio.sleep(1.0)

    def _parse_binance_message(self, data: dict[str, Any], received_at: datetime) -> RawMarketEvent | None:
        """Convert a raw Binance JSON message into a normalized RawMarketEvent."""
        event_type_str = data.get("e")

        if event_type_str == "trade":
            return RawMarketEvent(
                provider="binance",
                symbol=data.get("s", "UNKNOWN").upper(),
                channel="trade",
                received_at=received_at,
                raw_payload=data,
                sequence=data.get("t"),
            )
        elif event_type_str == "kline":
            return RawMarketEvent(
                provider="binance",
                symbol=data.get("s", "UNKNOWN").upper(),
                channel="kline",
                received_at=received_at,
                raw_payload=data,
                sequence=data.get("k", {}).get("t"),
            )
        elif event_type_str == "24hrTicker":
            return RawMarketEvent(
                provider="binance",
                symbol=data.get("s", "UNKNOWN").upper(),
                channel="ticker",
                received_at=received_at,
                raw_payload=data,
            )
        elif "u" in data and "b" in data and "a" in data:
            return RawMarketEvent(
                provider="binance",
                symbol=data.get("s", "UNKNOWN").upper(),
                channel="bookTicker",
                received_at=received_at,
                raw_payload=data,
                sequence=data.get("u"),
            )
        return None

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Candle]:
        """Fetch historical OHLCV candles from Binance REST API."""
        clean_symbol = symbol.replace("-", "").replace("/", "").upper()
        url = f"{self.rest_url}/klines"
        params = {
            "symbol": clean_symbol,
            "interval": timeframe,
            "startTime": int(start_time.timestamp() * 1000),
            "endTime": int(end_time.timestamp() * 1000),
            "limit": 1000,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        # Binance klines format: [ [ openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, ... ] ]
        candles: list[Candle] = []
        for row in data:
            c_time = datetime.fromtimestamp(row[0] / 1000.0, tz=UTC)
            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=c_time,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[7]) if len(row) > 7 else float(row[4]) * float(row[5]),
                    is_closed=True,
                )
            )
        return candles
