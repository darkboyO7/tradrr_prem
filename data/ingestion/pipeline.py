"""Trading Intelligence Platform - Real-Time Market Data Ingestion Pipeline.

Glues together Provider Acquisition -> Raw Storage -> Validation -> Normalization ->
Candle Construction -> Database Persistence -> Pub/Sub Broadcasting.
"""

import asyncio
import time
from typing import Any

from core.cache import cache
from core.database import get_db_session
from core.events import Candle, NormalizedTradeEvent, RawMarketEvent
from core.logging import get_logger
from data.candle.engine import CandleEngine
from data.ingestion.base import MarketDataProvider
from data.normalization.normalizer import EventNormalizer
from data.storage.repository import MarketDataRepository
from data.validation.validator import DataValidator

logger = get_logger(__name__)


class LiveIngestionPipeline:
    """Full-lifecycle streaming pipeline coordinator for high-frequency market data."""

    def __init__(
        self,
        provider: MarketDataProvider,
        validator: DataValidator | None = None,
        normalizer: EventNormalizer | None = None,
        candle_engine: CandleEngine | None = None,
        session_factory: Any | None = None,
    ) -> None:
        self.provider = provider
        self.validator = validator or DataValidator()
        self.normalizer = normalizer or EventNormalizer()
        self.candle_engine = candle_engine or CandleEngine(timeframe="1m")
        self._session_factory = session_factory or get_db_session

        self._is_running: bool = False
        self._consumer_task: asyncio.Task[None] | None = None

        # Operational metrics counters
        self.raw_events_received: int = 0
        self.trades_normalized: int = 0
        self.candles_generated: int = 0
        self.anomalies_detected: int = 0
        self._start_time: float | None = None

    async def start(self) -> None:
        """Start the provider and launch the consumption and aggregation loop."""
        if self._is_running:
            return

        self._is_running = True
        self._start_time = time.monotonic()
        await self.provider.connect()
        self._consumer_task = asyncio.create_task(self._process_stream())
        logger.info("Live ingestion pipeline started", provider=self.provider.name)

    async def stop(self) -> None:
        """Stop processing, flush active candles to storage, and disconnect provider."""
        self._is_running = False
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        await self.provider.disconnect()

        # Flush any remaining in-progress candles
        flushed_candles = self.candle_engine.force_close_all()
        if flushed_candles:
            try:
                async with self._session_factory() as session:
                    repo = MarketDataRepository(session)
                    await repo.save_candles_batch(flushed_candles)
                    self.candles_generated += len(flushed_candles)
                    logger.info("Flushed active candles on pipeline shutdown", count=len(flushed_candles))
            except Exception as exc:
                logger.error("Error persisting flushed candles on shutdown", error=str(exc))

        logger.info("Live ingestion pipeline stopped", provider=self.provider.name)

    async def _process_stream(self) -> None:
        """Core streaming loop consuming raw events from provider."""
        batch_events: list[RawMarketEvent] = []
        last_batch_time = time.monotonic()

        while self._is_running:
            try:
                async for raw_event in self.provider.stream_raw_events():
                    if not self._is_running:
                        break

                    self.raw_events_received += 1
                    batch_events.append(raw_event)

                    # 1. Validate raw event structure
                    is_valid_raw, raw_anomalies = self.validator.validate_raw_event(raw_event)
                    if not is_valid_raw:
                        self.anomalies_detected += len(raw_anomalies)
                        continue

                    # 2. Normalize event
                    normalized = self.normalizer.normalize(raw_event)
                    if normalized is None:
                        continue

                    # Handle trade events
                    if isinstance(normalized, NormalizedTradeEvent):
                        # 3. Validate normalized trade
                        is_valid_trade, trade_anomalies = self.validator.validate_trade(normalized)
                        if trade_anomalies:
                            self.anomalies_detected += len(trade_anomalies)

                        if is_valid_trade:
                            self.trades_normalized += 1

                            # 4. Ingest into candle engine
                            closed_candles = self.candle_engine.process_trade(normalized)

                            # If any 1m bar completed, persist to database and broadcast to bus
                            if closed_candles:
                                await self._persist_closed_candles(closed_candles)

                            # Broadcast trade tick to cache / bus
                            await cache.publish(f"trades:{normalized.symbol}", normalized.model_dump_json())

                    # Periodic batch persistence of raw events (immutable audit store)
                    now = time.monotonic()
                    if len(batch_events) >= 50 or (now - last_batch_time) > 2.0:
                        if batch_events:
                            await self._persist_raw_events_batch(batch_events)
                            batch_events = []
                            last_batch_time = now

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in ingestion pipeline processing loop", error=str(exc))
                await asyncio.sleep(0.1)

    async def _persist_raw_events_batch(self, events: list[RawMarketEvent]) -> None:
        """Bulk save raw events to database."""
        try:
            async with self._session_factory() as session:
                repo = MarketDataRepository(session)
                await repo.save_raw_events_batch(events)
        except Exception as exc:
            logger.error("Failed to save raw events batch to database", error=str(exc))

    async def _persist_closed_candles(self, candles: list[Candle]) -> None:
        """Validate and store finalized candles in database and publish to message bus."""
        valid_candles: list[Candle] = []
        for c in candles:
            is_valid, candle_anomalies = self.validator.validate_candle(c)
            if not is_valid:
                self.anomalies_detected += len(candle_anomalies)
            else:
                valid_candles.append(c)

        if not valid_candles:
            return

        try:
            async with self._session_factory() as session:
                repo = MarketDataRepository(session)
                await repo.save_candles_batch(valid_candles)
                self.candles_generated += len(valid_candles)

                for vc in valid_candles:
                    logger.info(
                        "1-Minute Candle Finalized & Stored",
                        symbol=vc.symbol,
                        time=vc.timestamp.isoformat(),
                        open=vc.open,
                        high=vc.high,
                        low=vc.low,
                        close=vc.close,
                        volume=vc.volume,
                        trades=vc.trade_count,
                        vwap=vc.vwap
                    )
                    await cache.publish(f"candles:{vc.symbol}:{vc.timeframe}", vc.model_dump_json())
        except Exception as exc:
            logger.error("Failed to persist finalized candles", error=str(exc))

    def get_metrics(self) -> dict[str, Any]:
        """Return operational pipeline statistics and latency profiles."""
        uptime = (time.monotonic() - self._start_time) if self._start_time else 0.0
        latencies = self.provider.get_latency_metrics()
        return {
            "provider": self.provider.name,
            "provider_state": self.provider.state.value,
            "uptime_seconds": round(uptime, 2),
            "raw_events_received": self.raw_events_received,
            "trades_normalized": self.trades_normalized,
            "candles_generated": self.candles_generated,
            "anomalies_detected": self.anomalies_detected,
            "latency": latencies,
        }
