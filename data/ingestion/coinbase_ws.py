"""Trading Intelligence Platform - Coinbase Public WebSocket Market Data Provider.

Streams live trades, quotes, and heartbeat events from Coinbase Exchange public WebSocket
with automatic reconnection, latency measurements, and historical candle backfilling.
"""

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from core.config import settings
from core.events import Candle, ProviderState, RawMarketEvent
from core.logging import get_logger
from data.ingestion.base import MarketDataProvider

logger = get_logger(__name__)

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
COINBASE_REST_URL = "https://api.exchange.coinbase.com"


class CoinbaseWebSocketProvider(MarketDataProvider):
    """Production Coinbase WebSocket provider supporting live trade streams and auto-reconnect."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        channels: list[str] | None = None,
        stale_threshold_seconds: float = 15.0,
        ws_url: str = COINBASE_WS_URL,
    ) -> None:
        super().__init__(name="coinbase_ws", stale_threshold_seconds=stale_threshold_seconds)
        self.ws_url = ws_url
        self._symbols = symbols or ["BTC-USD", "ETH-USD"]
        self._channels = channels or ["matches", "heartbeat"]
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._is_running: bool = False
        self._event_queue: asyncio.Queue[RawMarketEvent] = asyncio.Queue(maxsize=10000)
        self._worker_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Initiate background streaming worker and watchdog."""
        if self._is_running:
            return

        self._is_running = True
        self.set_state(ProviderState.CONNECTING)
        self.start_watchdog()
        self._worker_task = asyncio.create_task(self._connection_loop())
        logger.info("Coinbase WebSocket provider initiated", symbols=self._symbols, channels=self._channels)

    async def disconnect(self) -> None:
        """Gracefully disconnect WebSocket and terminate worker."""
        self._is_running = False
        self.stop_watchdog()
        if self._ws:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("Error closing websocket connection", error=str(exc))
            self._ws = None

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        self.set_state(ProviderState.DISCONNECTED)
        logger.info("Coinbase WebSocket provider disconnected")

    async def subscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Subscribe to additional symbols or channels dynamically."""
        for s in symbols:
            self._symbols.append(s)
        self._symbols = list(set(self._symbols))

        if self._ws and self._state == ProviderState.CONNECTED:
            sub_msg = {
                "type": "subscribe",
                "product_ids": symbols,
                "channels": channels,
            }
            await self._ws.send(json.dumps(sub_msg))
            logger.info("Sent dynamic subscription to Coinbase WS", product_ids=symbols, channels=channels)

    async def unsubscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Unsubscribe from symbols or channels."""
        for s in symbols:
            if s in self._symbols:
                self._symbols.remove(s)

        if self._ws and self._state == ProviderState.CONNECTED:
            unsub_msg = {
                "type": "unsubscribe",
                "product_ids": symbols,
                "channels": channels,
            }
            await self._ws.send(json.dumps(unsub_msg))
            logger.info("Sent unsubscribe message to Coinbase WS", product_ids=symbols)

    async def _connection_loop(self) -> None:
        """Resilient connection loop featuring exponential backoff with jitter."""
        attempt = 0
        backoff = settings.WS_RECONNECT_INITIAL_DELAY_SECONDS

        while self._is_running:
            try:
                self.set_state(ProviderState.CONNECTING if attempt == 0 else ProviderState.RECONNECTING)
                logger.info("Connecting to Coinbase WebSocket...", url=self.ws_url, attempt=attempt)

                async with websockets.connect(
                    self.ws_url,
                    ping_interval=settings.PING_INTERVAL_SECONDS,
                    ping_timeout=settings.PING_TIMEOUT_SECONDS,
                    close_timeout=5.0,
                ) as ws:
                    self._ws = ws
                    self.set_state(ProviderState.CONNECTED)
                    attempt = 0
                    backoff = settings.WS_RECONNECT_INITIAL_DELAY_SECONDS

                    # Send subscribe message
                    sub_payload = {
                        "type": "subscribe",
                        "product_ids": self._symbols,
                        "channels": self._channels,
                    }
                    await ws.send(json.dumps(sub_payload))
                    logger.info("Subscribed to Coinbase channels successfully", symbols=self._symbols)

                    # Ingest loop
                    async for raw_message in ws:
                        if not self._is_running:
                            break
                        self.touch_event()
                        await self._process_raw_message(raw_message)

            except ConnectionClosed as cc:
                logger.warning("Coinbase WebSocket connection closed by peer", code=cc.code, reason=cc.reason)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Coinbase WebSocket network error", error=str(exc))

            if self._is_running:
                self.set_state(ProviderState.RECONNECTING)
                attempt += 1
                # Exponential backoff with jitter
                sleep_time = min(backoff * (1.5 ** attempt) + random.uniform(0, 1.0), settings.WS_RECONNECT_MAX_DELAY_SECONDS)
                logger.info("Reconnecting to Coinbase in seconds", delay_secs=round(sleep_time, 2), attempt=attempt)
                await asyncio.sleep(sleep_time)

    async def _process_raw_message(self, message_str: str) -> None:
        """Parse raw incoming string and push to immutable raw event queue."""
        receive_time = datetime.now(UTC)
        try:
            payload: dict[str, Any] = json.loads(message_str)
            msg_type = payload.get("type", "")

            # Filter unneeded handshake/ack frames
            if msg_type in ("subscriptions", "error"):
                if msg_type == "error":
                    logger.error("Coinbase WS subscription error", message=payload.get("message"))
                return

            symbol = payload.get("product_id", "")
            seq = payload.get("sequence")

            # Calculate transit latency if exchange timestamp is present
            if "time" in payload and payload["time"]:
                try:
                    # e.g. 2026-10-04T08:00:00.123456Z
                    time_str = payload["time"].replace("Z", "+00:00")
                    exch_dt = datetime.fromisoformat(time_str)
                    latency = (receive_time - exch_dt).total_seconds() * 1000.0
                    if latency >= 0:
                        self.record_latency(latency)
                except Exception:
                    pass

            raw_event = RawMarketEvent(
                provider=self.name,
                symbol=symbol or "UNKNOWN",
                channel=msg_type,
                received_at=receive_time,
                raw_payload=payload,
                sequence=seq,
            )

            # Prevent queue blocking
            if self._event_queue.full():
                try:
                    self._event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self._event_queue.put(raw_event)

        except json.JSONDecodeError as exc:
            logger.error("Failed to decode JSON from Coinbase WS", error=str(exc))

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        """Async generator yielding incoming raw market events."""
        while self._is_running or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> list[Candle]:
        """Fetch historical candles from Coinbase REST API."""
        granularity_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400,
        }
        granularity = granularity_map.get(timeframe, 60)
        url = f"{COINBASE_REST_URL}/products/{symbol}/candles"
        params = {
            "granularity": granularity,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        # Coinbase candles format: [ [ time, low, high, open, close, volume ], ... ]
        candles: list[Candle] = []
        for row in reversed(data):
            c_time = datetime.fromtimestamp(row[0], UTC)
            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=c_time,
                    low=float(row[1]),
                    high=float(row[2]),
                    open=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[4]) * float(row[5]),
                    is_closed=True,
                )
            )
        return candles
