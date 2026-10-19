"""Trading Intelligence Platform - Data Storage Repository.

Provides transactional persistence and retrieval for raw immutable market events,
validated candles, data quality anomalies, and trade executions.
"""

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.events import Candle, DataQualityAnomaly, NormalizedTradeEvent, RawMarketEvent
from core.logging import get_logger
from data.storage.models import CandleModel, DataQualityEventModel, MarketEventModel, TradeModel

logger = get_logger(__name__)


class MarketDataRepository:
    """Async repository for market events, candles, and data quality records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save_raw_event(self, event: RawMarketEvent) -> MarketEventModel:
        """Persist an immutable raw market event."""
        model = MarketEventModel(
            provider=event.provider,
            symbol=event.symbol,
            channel=event.channel,
            received_at=event.received_at,
            raw_payload=event.raw_payload,
            sequence=event.sequence,
            created_at=datetime.now(UTC),
        )
        self.session.add(model)
        await self.session.flush()
        return model

    async def save_raw_events_batch(self, events: list[RawMarketEvent]) -> int:
        """Bulk insert raw market events."""
        if not events:
            return 0
        models = [
            MarketEventModel(
                provider=e.provider,
                symbol=e.symbol,
                channel=e.channel,
                received_at=e.received_at,
                raw_payload=e.raw_payload,
                sequence=e.sequence,
                created_at=datetime.now(UTC),
            )
            for e in events
        ]
        self.session.add_all(models)
        await self.session.flush()
        return len(models)

    async def save_candle(self, candle: Candle) -> CandleModel:
        """Persist or update a single OHLCV candle record."""
        # Check if candle already exists
        stmt = select(CandleModel).where(
            CandleModel.symbol == candle.symbol,
            CandleModel.timeframe == candle.timeframe,
            CandleModel.timestamp == candle.timestamp,
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            existing.open = candle.open
            existing.high = candle.high
            existing.low = candle.low
            existing.close = candle.close
            existing.volume = candle.volume
            existing.quote_volume = candle.quote_volume
            existing.trade_count = candle.trade_count
            existing.vwap = candle.vwap
            existing.is_closed = candle.is_closed
            await self.session.flush()
            return existing
        else:
            model = CandleModel(
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                quote_volume=candle.quote_volume,
                trade_count=candle.trade_count,
                vwap=candle.vwap,
                is_closed=candle.is_closed,
            )
            self.session.add(model)
            await self.session.flush()
            return model

    async def save_candles_batch(self, candles: list[Candle]) -> int:
        """Batch save candles."""
        count = 0
        for candle in candles:
            await self.save_candle(candle)
            count += 1
        return count

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[Candle]:
        """Fetch chronological candles for a symbol and timeframe."""
        stmt = select(CandleModel).where(
            CandleModel.symbol == symbol,
            CandleModel.timeframe == timeframe,
        )
        if start_time:
            stmt = stmt.where(CandleModel.timestamp >= start_time)
        if end_time:
            stmt = stmt.where(CandleModel.timestamp <= end_time)

        stmt = stmt.order_by(CandleModel.timestamp.asc()).limit(limit)
        result = await self.session.execute(stmt)
        records = result.scalars().all()

        return [
            Candle(
                symbol=r.symbol,
                timeframe=r.timeframe,
                timestamp=r.timestamp,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                quote_volume=r.quote_volume,
                trade_count=r.trade_count,
                vwap=r.vwap,
                is_closed=r.is_closed,
            )
            for r in records
        ]

    async def save_anomaly(self, anomaly: DataQualityAnomaly) -> DataQualityEventModel:
        """Persist a data quality anomaly event for audit and diagnostics."""
        model = DataQualityEventModel(
            provider=anomaly.provider,
            symbol=anomaly.symbol,
            check_name=anomaly.check_name,
            severity=anomaly.severity.value,
            description=anomaly.description,
            metrics=anomaly.metrics,
            timestamp=anomaly.timestamp,
        )
        self.session.add(model)
        await self.session.flush()
        return model

    async def save_trade(self, trade: NormalizedTradeEvent) -> TradeModel:
        """Persist an executed trade record."""
        model = TradeModel(
            trade_id=trade.trade_id,
            order_id=f"ext_{trade.trade_id}",
            symbol=trade.symbol,
            side=trade.side.value,
            price=trade.price,
            size=trade.size,
            fee=0.0,
            fee_currency="USD",
            executed_at=trade.timestamp,
            latency_ms=trade.latency_ms,
            is_simulated=False,
        )
        self.session.add(model)
        await self.session.flush()
        return model
