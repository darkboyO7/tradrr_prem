"""End-to-end Integration Test for the Ingestion Pipeline."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.events import ProviderState, RawMarketEvent
from data.candle.engine import CandleEngine
from data.ingestion.base import MarketDataProvider
from data.ingestion.pipeline import LiveIngestionPipeline
from data.normalization.normalizer import EventNormalizer
from data.storage.repository import MarketDataRepository
from data.validation.validator import DataValidator


class MockStreamProvider(MarketDataProvider):
    """Mock WebSocket provider yielding pre-determined chronological trade ticks."""

    def __init__(self, raw_events: list[RawMarketEvent]):
        super().__init__(name="mock_provider", stale_threshold_seconds=10.0)
        self._events = raw_events

    async def connect(self) -> None:
        self.set_state(ProviderState.CONNECTED)

    async def disconnect(self) -> None:
        self.set_state(ProviderState.DISCONNECTED)

    async def subscribe(self, symbols, channels) -> None:
        pass

    async def unsubscribe(self, symbols, channels) -> None:
        pass

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        for e in self._events:
            self.touch_event()
            yield e

    async def get_historical_candles(self, symbol, timeframe, start_time, end_time):
        return []


@pytest.mark.asyncio
async def test_live_ingestion_pipeline_end_to_end(test_engine):
    # Setup test session factory
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    @asynccontextmanager
    async def test_db_session():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    base_minute = datetime.now(UTC).replace(second=0, microsecond=0) - timedelta(minutes=2)
    t0 = base_minute + timedelta(seconds=10)
    t1 = base_minute + timedelta(seconds=40)
    t2 = base_minute + timedelta(minutes=1, seconds=5)

    raw_events = [
        RawMarketEvent(
            provider="coinbase_ws",
            symbol="BTC-USD",
            channel="matches",
            received_at=t0,
            raw_payload={
                "type": "match",
                "trade_id": "101",
                "price": "60000.0",
                "size": "1.0",
                "side": "buy",
                "product_id": "BTC-USD",
                "time": t0.isoformat(),
            },
        ),
        RawMarketEvent(
            provider="coinbase_ws",
            symbol="BTC-USD",
            channel="matches",
            received_at=t1,
            raw_payload={
                "type": "match",
                "trade_id": "102",
                "price": "60500.0",
                "size": "2.0",
                "side": "buy",
                "product_id": "BTC-USD",
                "time": t1.isoformat(),
            },
        ),
        RawMarketEvent(
            provider="coinbase_ws",
            symbol="BTC-USD",
            channel="matches",
            received_at=t2,
            raw_payload={
                "type": "match",
                "trade_id": "103",
                "price": "59800.0",
                "size": "0.5",
                "side": "sell",
                "product_id": "BTC-USD",
                "time": t2.isoformat(),
            },
        ),
    ]

    mock_provider = MockStreamProvider(raw_events)
    validator = DataValidator()
    normalizer = EventNormalizer()
    candle_engine = CandleEngine(timeframe="1m")

    pipeline = LiveIngestionPipeline(
        provider=mock_provider,
        validator=validator,
        normalizer=normalizer,
        candle_engine=candle_engine,
        session_factory=test_db_session,
    )

    await pipeline.start()
    # Allow stream to be consumed
    await asyncio.sleep(0.3)
    await pipeline.stop()

    # Query saved candles from database repository
    async with test_db_session() as session:
        repo = MarketDataRepository(session)
        candles = await repo.get_candles(symbol="BTC-USD", timeframe="1m")

        # Minute 0 was closed when minute 1 arrived, and minute 1 was flushed on stop
        assert len(candles) >= 1
        first_candle = candles[0]
        assert first_candle.symbol == "BTC-USD"
        assert first_candle.open == 60000.0
        assert first_candle.high == 60500.0
        assert first_candle.low == 60000.0
        assert first_candle.close == 60500.0
        assert first_candle.volume == 3.0
        assert first_candle.trade_count == 2
