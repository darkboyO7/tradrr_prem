"""Trading Intelligence Platform - BSE (Bombay Stock Exchange) Provider.

Institutional market data adapter for BSE equities supporting scrip codes,
IST market session boundaries, tick precision, and historical data retrieval.
"""

import urllib.parse
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from core.events import Candle, ProviderState, RawMarketEvent
from core.instrument import MarketType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from data.ingestion.base import MarketDataProvider
from data.ingestion.nse import _get_http_client

logger = get_logger(__name__)


class BSEDataProvider(MarketDataProvider):
    """Institutional Market Data Provider for Bombay Stock Exchange (BSE)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        stale_threshold_seconds: float = 30.0,
    ) -> None:
        super().__init__(name="bse_provider", stale_threshold_seconds=stale_threshold_seconds)
        self.api_key = api_key
        self.api_secret = api_secret
        self.market_type = MarketType.BSE
        self.session_info = SymbolResolver.get_session(MarketType.BSE)
        self.ist_tz = ZoneInfo("Asia/Kolkata")
        self._raw_queue: list[RawMarketEvent] = []

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Check if BSE trading session is active (09:15 - 15:30 IST Mon-Fri)."""
        if self.session_info:
            return self.session_info.is_open(dt)
        eval_dt = dt or datetime.now(UTC)
        ist_now = eval_dt.astimezone(self.ist_tz)
        if ist_now.weekday() >= 5:
            return False
        return time(9, 15) <= ist_now.time() <= time(15, 30)

    async def connect(self) -> None:
        """Connect to BSE data stream."""
        self.set_state(ProviderState.CONNECTING)
        logger.info("Connecting to BSE market data gateway", provider=self.name)
        self.set_state(ProviderState.CONNECTED)
        self.start_watchdog()
        self.touch_event()

    async def disconnect(self) -> None:
        """Disconnect from BSE data gateway."""
        self.stop_watchdog()
        self.set_state(ProviderState.DISCONNECTED)
        logger.info("Disconnected from BSE market data gateway", provider=self.name)

    async def subscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Subscribe to BSE equities / scrip codes."""
        for sym in symbols:
            resolved = SymbolResolver.resolve(sym)
            canonical = resolved.symbol if resolved else sym
            self._subscribed_symbols.add(canonical)
        for ch in channels:
            self._subscribed_channels.add(ch)

    async def unsubscribe(self, symbols: list[str], channels: list[str]) -> None:
        for sym in symbols:
            resolved = SymbolResolver.resolve(sym)
            canonical = resolved.symbol if resolved else sym
            self._subscribed_symbols.discard(canonical)
        for ch in channels:
            self._subscribed_channels.discard(ch)

    def push_mock_tick_for_test(self, symbol: str, price: float, volume: float) -> None:
        """Internal helper for deterministic offline/replay unit testing only."""
        resolved = SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol
        raw_event = RawMarketEvent(
            provider=self.name,
            symbol=clean_sym,
            channel="quote",
            received_at=datetime.now(UTC),
            raw_payload={
                "type": "quote",
                "exchange": "BSE",
                "symbol": clean_sym,
                "price": price,
                "volume": volume,
                "bid": round(price - 0.05, 2),
                "ask": round(price + 0.05, 2),
                "bid_qty": 500,
                "ask_qty": 500,
                "timestamp": datetime.now(UTC).timestamp(),
            },
        )
        self._raw_queue.append(raw_event)
        self.touch_event()

    def fetch_live_quote_sync(self, symbol: str, timeout: float = 6.0) -> RawMarketEvent | None:
        """Fetch real-time BSE quote via TradingView India Scanner or Groww BSE feed."""
        bse_query = symbol if ":" in symbol else f"BSE:{symbol.strip().upper()}"
        resolved = SymbolResolver.resolve(bse_query) or SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol.replace("BSE:", "").strip().upper()
        ticker = f"BSE:{clean_sym}"
        client = _get_http_client()

        try:
            resp = client.post(
                "https://scanner.tradingview.com/india/scan",
                json={
                    "symbols": {"tickers": [ticker, f"NSE:{clean_sym}"]},
                    "columns": ["close", "open", "high", "low", "volume", "change", "ATR"],
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("data", [])
                if rows and rows[0].get("d"):
                    d = rows[0]["d"]
                    price = float(d[0]) if d[0] is not None else 0.0
                    if price > 0:
                        open_p = float(d[1]) if d[1] is not None else price
                        high_p = float(d[2]) if d[2] is not None else price
                        low_p = float(d[3]) if d[3] is not None else price
                        vol = float(d[4]) if d[4] is not None else 0.0
                        atr_val = float(d[6]) if len(d) > 6 and d[6] is not None else 0.0
                        tick = resolved.tick_size if resolved else 0.05
                        half_spread = max(tick, round(price * 0.0001, 2))
                        now_dt = datetime.now(UTC)
                        raw_event = RawMarketEvent(
                            provider=self.name,
                            symbol=clean_sym,
                            channel="quote",
                            received_at=now_dt,
                            raw_payload={
                                "type": "quote",
                                "exchange": "BSE",
                                "symbol": clean_sym,
                                "price": price,
                                "open": open_p,
                                "high": high_p,
                                "low": low_p,
                                "volume": vol,
                                "atr": atr_val,
                                "bid": round(price - half_spread, 2),
                                "ask": round(price + half_spread, 2),
                                "bid_qty": 200,
                                "ask_qty": 200,
                                "timestamp": now_dt.timestamp(),
                                "source": "TRADINGVIEW_BSE_LIVE",
                            },
                        )
                        self._raw_queue.append(raw_event)
                        self.touch_event()
                        return raw_event
        except Exception as exc:
            logger.debug("TradingView BSE live quote fallback triggered", symbol=clean_sym, error=str(exc))

        candles = self.fetch_historical_candles_sync(bse_query, timeframe="5m", limit=5, timeout=timeout)
        if candles:
            last_c = candles[-1]
            price = last_c.close
            tick = resolved.tick_size if resolved else 0.05
            now_dt = datetime.now(UTC)
            raw_event = RawMarketEvent(
                provider=self.name,
                symbol=clean_sym,
                channel="quote",
                received_at=now_dt,
                raw_payload={
                    "type": "quote",
                    "exchange": "BSE",
                    "symbol": clean_sym,
                    "price": price,
                    "open": last_c.open,
                    "high": last_c.high,
                    "low": last_c.low,
                    "volume": last_c.volume,
                    "bid": round(price - tick, 2),
                    "ask": round(price + tick, 2),
                    "bid_qty": 100,
                    "ask_qty": 100,
                    "timestamp": now_dt.timestamp(),
                    "source": "GROWW_BSE_CASH",
                },
            )
            self._raw_queue.append(raw_event)
            self.touch_event()
            return raw_event

        return None

    def fetch_historical_candles_sync(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 240,
        timeout: float = 6.0,
    ) -> list[Candle]:
        """Fetch real BSE historical candles via Groww BSE Scrip or NSE cross-listed feed."""
        bse_query = symbol if ":" in symbol else f"BSE:{symbol.strip().upper()}"
        resolved = SymbolResolver.resolve(bse_query) or SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol.replace("BSE:", "").strip().upper()
        scrip_code = ""
        if resolved:
            scrip_code = str(resolved.metadata.get("scrip_code") or resolved.metadata.get("bse_scrip_code") or "")
        client = _get_http_client()

        tf_map = {
            "1m": ("monthly", 1),
            "5m": ("monthly", 5),
            "15m": ("monthly", 15),
            "1h": ("monthly", 60),
            "4h": ("monthly", 60),
            "1d": ("1y", 1440),
        }
        primary_spec = tf_map.get(timeframe, ("monthly", 5))
        specs_to_try = [primary_spec]
        if primary_spec != ("1y", 1440):
            specs_to_try.append(("1y", 1440))

        candles: list[Candle] = []
        for range_str, interval_min in specs_to_try:
            candidate_urls = []
            if scrip_code:
                candidate_urls.append(
                    f"https://groww.in/v1/api/charting_service/v2/chart/exchange/BSE/segment/CASH/"
                    f"{urllib.parse.quote(scrip_code)}/{range_str}?intervalInMinutes={interval_min}"
                )
            candidate_urls.append(
                f"https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/"
                f"{urllib.parse.quote(clean_sym)}/{range_str}?intervalInMinutes={interval_min}"
            )

            for url in candidate_urls:
                try:
                    resp = client.get(url, timeout=timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_candles = data.get("candles", [])
                        batch: list[Candle] = []
                        for row in raw_candles[-limit:]:
                            if len(row) >= 5 and row[1] is not None and row[4] is not None:
                                ts = datetime.fromtimestamp(float(row[0]), tz=UTC)
                                o_p = float(row[1])
                                h_p = float(row[2]) if row[2] is not None else o_p
                                l_p = float(row[3]) if row[3] is not None else o_p
                                c_p = float(row[4])
                                vol = float(row[5]) if len(row) > 5 and row[5] is not None else 0.0
                                if o_p > 0 and c_p > 0:
                                    batch.append(
                                        Candle(
                                            symbol=clean_sym,
                                            timeframe=timeframe,
                                            timestamp=ts,
                                            open=o_p,
                                            high=max(o_p, h_p, c_p),
                                            low=min(o_p, l_p, c_p),
                                            close=c_p,
                                            volume=max(0.0, vol),
                                        )
                                    )
                        if len(batch) >= 20:
                            self.touch_event()
                            return batch
                        elif len(batch) > len(candles):
                            candles = batch
                except Exception as exc:
                    logger.debug("BSE candle fetch attempt failed", url=url, error=str(exc))

        return candles

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        import asyncio

        while self.state == ProviderState.CONNECTED:
            if self._raw_queue:
                event = self._raw_queue.pop(0)
                yield event
            else:
                await asyncio.sleep(0.01)

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Candle]:
        """Fetch real historical BSE candles."""
        import asyncio

        return await asyncio.to_thread(
            self.fetch_historical_candles_sync,
            symbol=symbol,
            timeframe=timeframe,
            limit=240,
        )
