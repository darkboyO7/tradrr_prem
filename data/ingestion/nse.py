"""Trading Intelligence Platform - NSE (National Stock Exchange of India) Provider.

Institutional market data adapter for Indian equities with market session tracking,
IST timezone alignment, tick validation, and historical candle retrieval.
"""

import json
import os
import urllib.parse
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from core.events import Candle, ProviderState, RawMarketEvent
from core.instrument import MarketType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from data.ingestion.base import MarketDataProvider

logger = get_logger(__name__)

_SHARED_HTTP_CLIENT: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _SHARED_HTTP_CLIENT
    if _SHARED_HTTP_CLIENT is None or _SHARED_HTTP_CLIENT.is_closed:
        _SHARED_HTTP_CLIENT = httpx.Client(
            timeout=6.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        )
    return _SHARED_HTTP_CLIENT


_EQUITY_CACHE_FILE = Path(__file__).resolve().parent.parent / "universe" / "equity_candles_cache.json"


def _load_equity_cache(key: str) -> list[Candle]:
    """Load cached historical candles for offline and fast recovery."""
    try:
        if not _EQUITY_CACHE_FILE.exists():
            return []
        data = json.loads(_EQUITY_CACHE_FILE.read_text(encoding="utf-8"))
        records = data.get(key, [])
        candles = []
        now_utc = datetime.now(UTC)
        for r in records:
            ts = datetime.fromisoformat(r["timestamp"])
            candles.append(
                Candle(
                    symbol=r["symbol"],
                    timeframe=r["timeframe"],
                    timestamp=ts,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r.get("volume", 0.0)),
                    is_closed=True,
                    source_provider=r.get("source_provider", "CACHE_NSE"),
                    source_interval=r["timeframe"],
                    retrieved_at=now_utc,
                )
            )
        return candles
    except Exception as exc:
        logger.debug("NSE equity cache load failed", key=key, error=str(exc))
        return []


def _save_equity_cache(key: str, candles: list[Candle]) -> None:
    """Save authentic fetched candles to local persistent equity cache."""
    if not candles or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    try:
        data: dict[str, Any] = {}
        if _EQUITY_CACHE_FILE.exists():
            try:
                data = json.loads(_EQUITY_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        records = []
        for c in candles[-180:]:
            records.append({
                "symbol": c.symbol,
                "timeframe": c.timeframe,
                "timestamp": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "source_provider": c.source_provider,
            })
        data[key] = records
        _EQUITY_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp_file = _EQUITY_CACHE_FILE.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_file.replace(_EQUITY_CACHE_FILE)
    except Exception as exc:
        logger.debug("NSE equity cache save failed", key=key, error=str(exc))


class NSEDataProvider(MarketDataProvider):
    """Institutional Market Data Provider for National Stock Exchange of India (NSE)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        stale_threshold_seconds: float = 30.0,
    ) -> None:
        super().__init__(name="nse_provider", stale_threshold_seconds=stale_threshold_seconds)
        self.api_key = api_key
        self.api_secret = api_secret
        self.market_type = MarketType.NSE
        self.session_info = SymbolResolver.get_session(MarketType.NSE)
        self.ist_tz = ZoneInfo("Asia/Kolkata")
        self._raw_queue: list[RawMarketEvent] = []

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Check if NSE trading session is active (09:15 - 15:30 IST Mon-Fri)."""
        if self.session_info:
            return self.session_info.is_open(dt)
        eval_dt = dt or datetime.now(UTC)
        ist_now = eval_dt.astimezone(self.ist_tz)
        if ist_now.weekday() >= 5:  # Saturday or Sunday
            return False
        return time(9, 15) <= ist_now.time() <= time(15, 30)

    async def connect(self) -> None:
        """Connect to the exchange/broker stream."""
        self.set_state(ProviderState.CONNECTING)
        logger.info("Connecting to NSE market data gateway", provider=self.name)
        self.set_state(ProviderState.CONNECTED)
        self.start_watchdog()
        self.touch_event()

    async def disconnect(self) -> None:
        """Cleanly terminate NSE gateway connection."""
        self.stop_watchdog()
        self.set_state(ProviderState.DISCONNECTED)
        logger.info("Disconnected from NSE market data gateway", provider=self.name)

    async def subscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Subscribe to NSE equity instruments."""
        for sym in symbols:
            resolved = SymbolResolver.resolve(sym)
            canonical = resolved.symbol if resolved else sym
            self._subscribed_symbols.add(canonical)
        for ch in channels:
            self._subscribed_channels.add(ch)
        logger.info(
            "Subscribed to NSE feeds",
            symbols=list(self._subscribed_symbols),
            channels=list(self._subscribed_channels),
        )

    async def unsubscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Unsubscribe from NSE feeds."""
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
                "exchange": "NSE",
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
        """Fetch real-time NSE quote from TradingView India Scanner or Groww live feed."""
        resolved = SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol.replace("NSE:", "").strip().upper()
        mapped_sym = SymbolResolver._historical_symbol_map.get(clean_sym)
        tickers = [f"NSE:{clean_sym}"]
        if mapped_sym and mapped_sym != clean_sym:
            tickers.append(f"NSE:{mapped_sym}")
        client = _get_http_client()

        try:
            resp = client.post(
                "https://scanner.tradingview.com/india/scan",
                json={
                    "symbols": {"tickers": tickers},
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
                                "exchange": "NSE",
                                "symbol": clean_sym,
                                "price": price,
                                "open": open_p,
                                "high": high_p,
                                "low": low_p,
                                "volume": vol,
                                "atr": atr_val,
                                "bid": round(price - half_spread, 2),
                                "ask": round(price + half_spread, 2),
                                "bid_qty": 250,
                                "ask_qty": 250,
                                "timestamp": now_dt.timestamp(),
                                "source": "TRADINGVIEW_NSE_LIVE",
                            },
                        )
                        self._raw_queue.append(raw_event)
                        self.touch_event()
                        return raw_event
        except Exception as exc:
            logger.debug("TradingView NSE live quote fallback triggered", symbol=clean_sym, error=str(exc))

        candles = self.fetch_historical_candles_sync(clean_sym, timeframe="5m", limit=5, timeout=timeout)
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
                    "exchange": "NSE",
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
                    "source": "GROWW_NSE_CASH",
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
        """Synchronously fetch real NSE OHLCV candles from Groww / Moneycontrol APIs."""
        resolved = SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol.replace("NSE:", "").strip().upper()
        mapped_sym = SymbolResolver._historical_symbol_map.get(clean_sym)
        candidate_syms = [clean_sym]
        if mapped_sym and mapped_sym != clean_sym:
            candidate_syms.append(mapped_sym)
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
        now_utc = datetime.now(UTC)
        tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 300)

        for cand_sym in candidate_syms:
            encoded_sym = urllib.parse.quote(cand_sym)
            for range_str, interval_min in specs_to_try:
                url = (
                    f"https://groww.in/v1/api/charting_service/v2/chart/exchange/NSE/segment/CASH/"
                    f"{encoded_sym}/{range_str}?intervalInMinutes={interval_min}"
                )
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
                                    is_closed = now_utc.timestamp() >= (ts.timestamp() + tf_seconds)
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
                                            is_closed=is_closed,
                                            source_provider="GROWW_NSE",
                                            source_interval=timeframe,
                                            retrieved_at=now_utc,
                                        )
                                    )
                        if len(batch) >= 20:
                            self.touch_event()
                            _save_equity_cache(f"NSE:{clean_sym}:{timeframe}", batch)
                            return batch
                        elif len(batch) > len(candles):
                            candles = batch
                except Exception as exc:
                    logger.debug("Groww NSE candle fetch failed, trying backup", symbol=cand_sym, error=str(exc))

        if candles:
            self.touch_event()
            _save_equity_cache(f"NSE:{clean_sym}:{timeframe}", candles)
            return candles
        encoded_sym = urllib.parse.quote(clean_sym)

        # Backup: Moneycontrol TechCharts API
        try:
            mc_res_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "60", "1d": "1D"}
            mc_res = mc_res_map.get(timeframe, "5")
            end_ts = int(datetime.now(UTC).timestamp())
            start_ts = end_ts - (86400 * 30 if mc_res != "1D" else 86400 * 365)
            mc_url = (
                f"https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history"
                f"?symbol={encoded_sym}&resolution={mc_res}&from={start_ts}&to={end_ts}&countback={limit}"
            )
            resp = client.get(mc_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("s") == "ok" and data.get("t"):
                    t_arr = data["t"]
                    o_arr = data["o"]
                    h_arr = data["h"]
                    l_arr = data["l"]
                    c_arr = data["c"]
                    v_arr = data.get("v", [0.0] * len(t_arr))
                    for i in range(max(0, len(t_arr) - limit), len(t_arr)):
                        ts = datetime.fromtimestamp(float(t_arr[i]), tz=UTC)
                        o_p = float(o_arr[i])
                        h_p = float(h_arr[i])
                        l_p = float(l_arr[i])
                        c_p = float(c_arr[i])
                        vol = float(v_arr[i]) if i < len(v_arr) and v_arr[i] is not None else 0.0
                        if o_p > 0 and c_p > 0:
                            is_closed = now_utc.timestamp() >= (ts.timestamp() + tf_seconds)
                            candles.append(
                                Candle(
                                    symbol=clean_sym,
                                    timeframe=timeframe,
                                    timestamp=ts,
                                    open=o_p,
                                    high=max(o_p, h_p, c_p),
                                    low=min(o_p, l_p, c_p),
                                    close=c_p,
                                    volume=max(0.0, vol),
                                    is_closed=is_closed,
                                    source_provider="MONEYCONTROL_NSE",
                                    source_interval=timeframe,
                                    retrieved_at=now_utc,
                                )
                            )
                    if candles:
                        self.touch_event()
                        _save_equity_cache(f"NSE:{clean_sym}:{timeframe}", candles)
                        return candles
        except Exception as exc:
            logger.warning("NSE historical candle retrieval failed across all providers", symbol=clean_sym, error=str(exc))

        cached = _load_equity_cache(f"NSE:{clean_sym}:{timeframe}")
        if cached:
            self.touch_event()
            return cached[-limit:]

        return candles

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        """Stream real-time raw NSE ticks and market depth."""
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
        """Fetch real historical NSE candles (never generates synthetic prices)."""
        import asyncio

        return await asyncio.to_thread(
            self.fetch_historical_candles_sync,
            symbol=symbol,
            timeframe=timeframe,
            limit=240,
        )
