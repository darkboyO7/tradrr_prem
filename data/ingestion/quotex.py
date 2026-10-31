"""Trading Intelligence Platform - Quotex Adapter & Underlying Resolution Engine.

Implements instrument discovery, separation of Platform Instrument -> Underlying Instrument ->
Market Data Source -> Prediction Instrument, and strict data-availability gating:
- For Quotex instruments backed by authorized live underlying feeds (FX Spot, FX OTC Underlying Proxy,
  Crypto Spot), fetches real market candles and live quotes from the underlying feed.
- For proprietary synthetic OTC instruments without an authorized public live feed (e.g. MSFT_OTC,
  AAPL_OTC), returns None / empty candles so the pipeline explicitly reports:
  'Quotex live data unavailable' rather than fabricating prices.
"""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from core.events import Candle, ProviderState, RawMarketEvent
from core.instrument import MarketType
from core.logging import get_logger
from core.symbol_resolver import SymbolResolver
from core.universe import QuotexDiscoveryAdapter, QuotexInstrumentAdapter
from data.ingestion.base import MarketDataProvider
from data.ingestion.forex import ForexDataProvider

logger = get_logger(__name__)


class QuotexMarketDataAdapter(MarketDataProvider):
    """Institutional Quotex Market Adapter separating Platform Instrument from Underlying Data Source.

    Connects verified underlying feeds (TradingView interbank FX and Coinbase/Kraken Crypto)
    for analysis and paper execution while honestly reporting UNAVAILABLE for proprietary broker synthetic OTC.
    """

    def __init__(
        self,
        api_key: str | None = None,
        stale_threshold_seconds: float = 20.0,
    ) -> None:
        super().__init__(name="quotex_provider", stale_threshold_seconds=stale_threshold_seconds)
        self.api_key = api_key
        self.market_type = MarketType.QUOTEX
        self._raw_queue: list[RawMarketEvent] = []
        self._fx_proxy = ForexDataProvider(stale_threshold_seconds=stale_threshold_seconds)

    def is_market_open(self, dt: datetime | None = None) -> bool:
        """Quotex markets operate 24/7; individual instrument availability is governed by QuotexInstrumentAdapter."""
        return True

    def is_live_data_available(self, symbol: str) -> bool:
        """Check whether the Quotex instrument has a legitimate authorized live market data source."""
        return QuotexInstrumentAdapter.is_live_data_available(symbol)


    async def connect(self) -> None:
        """Connect to authorized feed."""
        self.set_state(ProviderState.CONNECTING)
        logger.info("Connecting to Quotex paper data feed (Analysis Mode)", provider=self.name)
        self.set_state(ProviderState.CONNECTED)
        self.start_watchdog()
        self.touch_event()

    async def disconnect(self) -> None:
        self.stop_watchdog()
        self.set_state(ProviderState.DISCONNECTED)
        logger.info("Disconnected from Quotex paper feed", provider=self.name)

    async def subscribe(self, symbols: list[str], channels: list[str]) -> None:
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

    def push_mock_tick_for_test(self, symbol: str, price: float, payout: float = 0.85) -> None:
        resolved = SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol

        raw_event = RawMarketEvent(
            provider=self.name,
            symbol=clean_sym,
            channel="quote",
            received_at=datetime.now(UTC),
            raw_payload={
                "type": "quote",
                "exchange": "QUOTEX",
                "symbol": clean_sym,
                "price": price,
                "payout": payout,
                "timestamp": datetime.now(UTC).timestamp(),
            },
        )
        self._raw_queue.append(raw_event)
        self.touch_event()

    def fetch_live_quote_sync(self, symbol: str, timeout: float = 8.0) -> RawMarketEvent | None:
        """Fetch real underlying quote for Quotex analysis & paper mode if an authorized feed exists."""
        resolved = SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol.replace("QUOTEX:", "").strip().upper()
        rec = QuotexDiscoveryAdapter.get_record(clean_sym)

        if rec is not None and not rec.live_data_available:
            logger.info(
                "Quotex live data unavailable for proprietary synthetic OTC instrument",
                quotex_symbol=clean_sym,
                underlying=rec.underlying_instrument,
                data_source=rec.market_data_source,
            )
            return None

        underlying = rec.underlying_instrument if rec else f"FX:{clean_sym.replace('_OTC', '')}"
        payout = rec.payout_pct if rec else (float(resolved.metadata.get("payout_pct", 0.88)) if resolved else 0.85)

        if underlying.startswith("FX:"):
            underlying_fx = underlying.split(":", 1)[1]
            fx_ev = self._fx_proxy.fetch_live_quote_sync(underlying_fx, timeout=timeout)
            if fx_ev:
                price = float(fx_ev.raw_payload.get("price", fx_ev.raw_payload.get("mid", 0.0)))
                now_dt = datetime.now(UTC)
                raw_event = RawMarketEvent(
                    provider=self.name,
                    symbol=clean_sym,
                    channel="quote",
                    received_at=now_dt,
                    raw_payload={
                        "type": "quote",
                        "exchange": "QUOTEX",
                        "symbol": clean_sym,
                        "canonical_symbol": f"QUOTEX:{clean_sym}",
                        "display_name": rec.display_name if rec else clean_sym,
                        "underlying_instrument": underlying,
                        "underlying_market_symbol": rec.underlying_market_symbol if rec else f"FX_IDC:{underlying.split(':', 1)[1]}",
                        "underlying_provider": rec.underlying_provider if rec else "TradingView / Interbank FX",
                        "market_data_source": rec.market_data_source if rec else "AUTHORIZED_FX_IDC_UNDERLYING_FEED",
                        "data_quality_status": rec.data_quality_status if rec else "VERIFIED_INTERBANK",
                        "limitation_notes": rec.limitation_notes if rec else "",
                        "supported_horizons": rec.supported_horizons if rec else ["5m", "15m", "1h", "4h", "1d"],
                        "price": price,
                        "bid": float(fx_ev.raw_payload.get("bid", price)),
                        "ask": float(fx_ev.raw_payload.get("ask", price)),
                        "payout": payout,
                        "timestamp": now_dt.timestamp(),
                        "source": "QUOTEX_ANALYSIS_FX_UNDERLYING",
                    },
                )
                self._raw_queue.append(raw_event)
                self.touch_event()
                return raw_event

        elif underlying.startswith("CRYPTO:"):
            from data.market_state import MarketStateService

            und_inst = SymbolResolver.resolve(underlying)
            if und_inst:
                crypto_candles, crypto_quote = MarketStateService.get_instance()._fetch_crypto_candles_and_quote(
                    und_inst, timeframe="5m", limit=80
                )
                if crypto_quote:
                    bid = float(crypto_quote.raw_payload.get("bid", 0.0))
                    ask = float(crypto_quote.raw_payload.get("ask", bid))
                    price = round((bid + ask) / 2.0, 2) if (bid > 0 and ask > 0) else bid
                    ts_val = float(crypto_quote.raw_payload.get("time", datetime.now(UTC).timestamp()))
                elif crypto_candles:
                    latest = crypto_candles[-1]
                    price = float(latest.close)
                    bid = round(price * 0.9999, 2)
                    ask = round(price * 1.0001, 2)
                    ts_val = latest.timestamp.timestamp()
                else:
                    price = 0.0
                    bid = 0.0
                    ask = 0.0
                    ts_val = 0.0

                if price > 0:
                    now_dt = datetime.now(UTC)
                    raw_event = RawMarketEvent(
                        provider=self.name,
                        symbol=clean_sym,
                        channel="quote",
                        received_at=now_dt,
                        raw_payload={
                            "type": "quote",
                            "exchange": "QUOTEX",
                            "symbol": clean_sym,
                            "canonical_symbol": f"QUOTEX:{clean_sym}",
                            "display_name": rec.display_name if rec else clean_sym,
                            "underlying_instrument": underlying,
                            "underlying_market_symbol": rec.underlying_market_symbol if rec else "COINBASE:BTCUSD",
                            "underlying_provider": rec.underlying_provider if rec else "Coinbase / Kraken Spot",
                            "market_data_source": rec.market_data_source if rec else "AUTHORIZED_CRYPTO_UNDERLYING_FEED",
                            "data_quality_status": rec.data_quality_status if rec else "VERIFIED_CRYPTO",
                            "limitation_notes": rec.limitation_notes if rec else "",
                            "supported_horizons": rec.supported_horizons if rec else ["5m", "15m", "1h", "4h", "1d"],
                            "price": price,
                            "bid": bid,
                            "ask": ask,
                            "payout": payout,
                            "timestamp": ts_val or now_dt.timestamp(),
                            "source": "QUOTEX_ANALYSIS_CRYPTO_UNDERLYING",
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
        timeout: float = 10.0,
    ) -> list[Candle]:
        """Fetch real historical candles from the mapped underlying asset for Quotex analysis."""
        resolved = SymbolResolver.resolve(symbol)
        clean_sym = resolved.symbol if resolved else symbol.replace("QUOTEX:", "").strip().upper()
        rec = QuotexDiscoveryAdapter.get_record(clean_sym)

        if rec is not None and not rec.live_data_available:
            return []

        underlying = rec.underlying_instrument if rec else f"FX:{clean_sym.replace('_OTC', '')}"
        now_utc = datetime.now(UTC)
        tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 300)

        if underlying.startswith("FX:"):
            underlying_fx = underlying.split(":", 1)[1]
            fx_candles = self._fx_proxy.fetch_historical_candles_sync(
                underlying_fx,
                timeframe=timeframe,
                limit=limit,
                timeout=timeout,
            )
            return [
                Candle(
                    symbol=clean_sym,
                    timeframe=c.timeframe,
                    timestamp=c.timestamp,
                    open=c.open,
                    high=c.high,
                    low=c.low,
                    close=c.close,
                    volume=c.volume,
                    is_closed=now_utc.timestamp() >= (c.timestamp.timestamp() + tf_seconds),
                    source_provider="UNDERLYING_FX_INTERBANK_PROXY",
                    source_interval=c.timeframe,
                    retrieved_at=now_utc,
                )
                for c in fx_candles
            ]
        elif underlying.startswith("CRYPTO:"):
            from data.market_state import MarketStateService

            und_inst = SymbolResolver.resolve(underlying)
            if und_inst:
                c_list, _ = MarketStateService.get_instance()._fetch_crypto_candles_and_quote(
                    und_inst, timeframe=timeframe, limit=limit
                )
                return [
                    Candle(
                        symbol=clean_sym,
                        timeframe=c.timeframe,
                        timestamp=c.timestamp,
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume,
                        is_closed=now_utc.timestamp() >= (c.timestamp.timestamp() + tf_seconds),
                        source_provider="COINBASE_KRAKEN_CRYPTO_PROXY",
                        source_interval=c.timeframe,
                        retrieved_at=now_utc,
                    )
                    for c in c_list
                ]
        return []

    def fetch_multi_timeframe_candles_sync(
        self,
        symbol: str,
        timeframes: list[str] | None = None,
        limit: int = 120,
        timeout: float = 8.0,
    ) -> dict[str, list[Candle]]:
        """Fetch synchronized candles across multiple timeframes (5m, 15m, 1h, 4h, 1d) with zero lookahead."""
        tfs = timeframes or ["5m", "15m", "1h", "4h", "1d"]
        result: dict[str, list[Candle]] = {}
        for tf in tfs:
            result[tf] = self.fetch_historical_candles_sync(
                symbol=symbol,
                timeframe=tf,
                limit=limit,
                timeout=timeout,
            )
        return result

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
        import asyncio

        return await asyncio.to_thread(
            self.fetch_historical_candles_sync,
            symbol=symbol,
            timeframe=timeframe,
            limit=240,
        )


QuotexDataProvider = QuotexMarketDataAdapter

