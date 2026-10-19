"""Trading Intelligence Platform - Kraken Market Data Provider.

Supports streaming public trade and spread WebSocket feeds from Kraken,
normalizing Kraken-specific array payloads into canonical RawMarketEvent objects.
"""

import asyncio
from datetime import datetime, timezone
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
import websockets
from websockets.exceptions import ConnectionClosed

from core.events import Candle, ProviderState, RawMarketEvent
from core.logging import get_logger
from data.ingestion.base import MarketDataProvider

logger = get_logger("kraken_ws")


class KrakenWebSocketProvider(MarketDataProvider):
    """High-performance WebSocket connector for Kraken public market data."""

    BASE_WS_URL = "wss://ws.kraken.com"
    BASE_REST_URL = "https://api.kraken.com/0/public"

    # Map standard tickers to Kraken pair representations
    SYMBOL_MAP = {
        "BTC-USD": "XBT/USD",
        "BTC/USD": "XBT/USD",
        "ETH-USD": "ETH/USD",
        "ETH/USD": "ETH/USD",
        "SOL-USD": "SOL/USD",
        "SOL/USD": "SOL/USD",
    }

    REVERSE_SYMBOL_MAP = {
        "XBT/USD": "BTC-USD",
        "XXBTZUSD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "XETHZUSD": "ETH-USD",
        "SOL/USD": "SOL-USD",
    }

    def __init__(
        self,
        symbols: List[str],
        channels: Optional[List[str]] = None,
        stale_threshold_seconds: float = 15.0,
        ws_url: Optional[str] = None,
        rest_url: Optional[str] = None,
    ) -> None:
        super().__init__(name="kraken", stale_threshold_seconds=stale_threshold_seconds)
        self.raw_symbols = symbols
        self.kraken_pairs = [self.SYMBOL_MAP.get(s.upper(), s.upper()) for s in symbols]
        self.channels = channels or ["trade"]
        self.ws_url = ws_url or self.BASE_WS_URL
        self.rest_url = rest_url or self.BASE_REST_URL
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False

    async def connect(self) -> None:
        """Establish WebSocket connection with Kraken and subscribe to initial feeds."""
        self.set_state(ProviderState.CONNECTING)

        try:
            self._ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            )
            self._running = True
            self.set_state(ProviderState.CONNECTED)
            self.start_watchdog()

            # Subscribe to initial pairs and channels
            for ch in self.channels:
                sub_msg = {
                    "event": "subscribe",
                    "pair": self.kraken_pairs,
                    "subscription": {"name": ch},
                }
                await self._ws.send(json.dumps(sub_msg))

            logger.info("Connected to Kraken WebSocket", pairs=self.kraken_pairs, channels=self.channels)
        except Exception as e:
            self.set_state(ProviderState.FAILED)
            logger.error("Failed to connect to Kraken WebSocket", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Gracefully disconnect from Kraken WebSocket."""
        self._running = False
        self.stop_watchdog()
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.set_state(ProviderState.DISCONNECTED)
        logger.info("Disconnected from Kraken WebSocket")

    async def subscribe(self, symbols: List[str], channels: List[str]) -> None:
        """Dynamically subscribe to Kraken feeds."""
        pairs = [self.SYMBOL_MAP.get(s.upper(), s.upper()) for s in symbols]
        for p in pairs:
            self._subscribed_symbols.add(p)
        for c in channels:
            self._subscribed_channels.add(c)

        if not self._ws or not self._running:
            return

        for ch in channels:
            sub_msg = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": ch},
            }
            await self._ws.send(json.dumps(sub_msg))
            logger.info("Sent Kraken subscription request", pairs=pairs, channel=ch)

    async def unsubscribe(self, symbols: List[str], channels: List[str]) -> None:
        """Unsubscribe from Kraken feeds."""
        pairs = [self.SYMBOL_MAP.get(s.upper(), s.upper()) for s in symbols]
        if not self._ws or not self._running:
            return

        for ch in channels:
            unsub_msg = {
                "event": "unsubscribe",
                "pair": pairs,
                "subscription": {"name": ch},
            }
            await self._ws.send(json.dumps(unsub_msg))
            logger.info("Sent Kraken unsubscribe request", pairs=pairs, channel=ch)

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        """Continuously yield incoming RawMarketEvent objects from Kraken feed."""
        backoff_seconds = 1.0

        while self._running:
            try:
                if not self._ws:
                    await self.connect()

                assert self._ws is not None
                raw_msg = await self._ws.recv()
                receive_time = datetime.now(timezone.utc)

                data = json.loads(raw_msg)

                # Skip heartbeats and connection/subscription status acknowledgements
                if isinstance(data, dict):
                    if data.get("event") in ("heartbeat", "systemStatus", "subscriptionStatus"):
                        self.update_freshness()
                        continue

                events = self._parse_kraken_message(data, receive_time)
                for event in events:
                    self.update_freshness()
                    yield event

                backoff_seconds = 1.0

            except ConnectionClosed as e:
                logger.warning("Kraken connection closed, reconnecting...", error=str(e))
                self.set_state(ProviderState.RECONNECTING)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 30.0)
                try:
                    await self.connect()
                except Exception:
                    pass
            except Exception as e:
                logger.error("Error receiving from Kraken WebSocket", error=str(e))
                await asyncio.sleep(1.0)

    def _parse_kraken_message(self, data: Any, received_at: datetime) -> List[RawMarketEvent]:
        """Convert a raw Kraken array payload into normalized RawMarketEvent instances."""
        # Kraken publication messages arrive as arrays: [channelID, payload, channelName, pair]
        if not isinstance(data, list) or len(data) < 4:
            return []

        channel_name = data[2]
        pair = data[3]
        canonical_symbol = self.REVERSE_SYMBOL_MAP.get(pair, pair.replace("/", "-"))
        results = []

        if channel_name == "trade":
            # trade array: [ [ price, volume, time, side, orderType, misc ], ... ]
            trades = data[1]
            for t in trades:
                results.append(
                    RawMarketEvent(
                        provider="kraken",
                        symbol=canonical_symbol,
                        channel="trade",
                        received_at=received_at,
                        raw_payload={
                            "price": t[0],
                            "volume": t[1],
                            "time": t[2],
                            "side": t[3],  # "b" = buy, "s" = sell
                            "order_type": t[4],
                            "pair": pair,
                        },
                    )
                )
        elif channel_name == "spread":
            # spread: [ bid, ask, time, bidVolume, askVolume ]
            sp = data[1]
            results.append(
                RawMarketEvent(
                    provider="kraken",
                    symbol=canonical_symbol,
                    channel="spread",
                    received_at=received_at,
                    raw_payload={
                        "bid": sp[0],
                        "ask": sp[1],
                        "time": sp[2],
                        "bid_volume": sp[3],
                        "ask_volume": sp[4],
                        "pair": pair,
                    },
                )
            )

        return results

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Candle]:
        """Fetch historical candles from Kraken REST public OHLC endpoint."""
        pair = self.SYMBOL_MAP.get(symbol.upper(), symbol.upper().replace("-", "/"))
        url = f"{self.rest_url}/OHLC"

        # Timeframe mapping (Kraken uses minutes: 1, 5, 15, 60, 240, 1440)
        tf_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = tf_map.get(timeframe, 1)

        params = {
            "pair": pair,
            "interval": interval,
            "since": int(start_time.timestamp()),
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        if data.get("error"):
            logger.error("Kraken REST API error", errors=data["error"])
            return []

        result = data.get("result", {})
        # Kraken returns dict where key is pair name: {"XXBTZUSD": [ [time, open, high, low, close, vwap, volume, count], ... ], "last": ...}
        pair_key = next((k for k in result.keys() if k != "last"), None)
        if not pair_key:
            return []

        raw_candles = result[pair_key]
        candles: List[Candle] = []

        for row in raw_candles:
            ts = datetime.fromtimestamp(row[0], tz=timezone.utc)
            if ts < start_time or ts > end_time:
                continue
            candles.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[6]),
                    quote_volume=float(row[4]) * float(row[6]),
                    is_closed=True,
                )
            )

        return candles
