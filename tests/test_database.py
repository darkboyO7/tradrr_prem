"""Tests for Database Schema, Models, and Repository persistence."""

from datetime import UTC, datetime

import pytest

from core.events import (
    Candle,
    DataQualityAnomaly,
    DataQualitySeverity,
    NormalizedTradeEvent,
    RawMarketEvent,
    TradeSide,
)
from data.storage.repository import MarketDataRepository


@pytest.mark.asyncio
async def test_save_and_query_candles(db_session):
    repo = MarketDataRepository(db_session)
    t0 = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 10, 4, 12, 1, 0, tzinfo=UTC)

    candle1 = Candle(
        symbol="BTC-USD",
        timeframe="1m",
        timestamp=t0,
        open=65000.0,
        high=65100.0,
        low=64900.0,
        close=65050.0,
        volume=2.5,
        is_closed=True,
    )
    candle2 = Candle(
        symbol="BTC-USD",
        timeframe="1m",
        timestamp=t1,
        open=65050.0,
        high=65200.0,
        low=65000.0,
        close=65150.0,
        volume=3.1,
        is_closed=True,
    )

    await repo.save_candles_batch([candle1, candle2])
    await db_session.commit()

    queried = await repo.get_candles(symbol="BTC-USD", timeframe="1m")
    assert len(queried) == 2
    assert queried[0].open == 65000.0
    assert queried[1].close == 65150.0


@pytest.mark.asyncio
async def test_save_raw_events_batch(db_session):
    repo = MarketDataRepository(db_session)
    now = datetime.now(UTC)
    events = [
        RawMarketEvent(
            provider="coinbase_ws",
            symbol="BTC-USD",
            channel="matches",
            received_at=now,
            raw_payload={"trade_id": 1, "price": "65000"},
            sequence=1,
        ),
        RawMarketEvent(
            provider="coinbase_ws",
            symbol="BTC-USD",
            channel="matches",
            received_at=now,
            raw_payload={"trade_id": 2, "price": "65010"},
            sequence=2,
        ),
    ]

    saved_count = await repo.save_raw_events_batch(events)
    await db_session.commit()
    assert saved_count == 2


@pytest.mark.asyncio
async def test_save_anomaly_and_trade(db_session):
    repo = MarketDataRepository(db_session)
    anomaly = DataQualityAnomaly(
        provider="coinbase_ws",
        symbol="SOL-USD",
        check_name="invalid_price",
        severity=DataQualitySeverity.ERROR,
        description="Negative price test",
        metrics={"price": -5.0},
    )
    anomaly_model = await repo.save_anomaly(anomaly)
    assert anomaly_model.id is not None
    assert anomaly_model.check_name == "invalid_price"

    trade = NormalizedTradeEvent(
        provider="coinbase_ws",
        symbol="SOL-USD",
        trade_id="SOL-999",
        timestamp=datetime.now(UTC),
        received_at=datetime.now(UTC),
        price=150.0,
        size=10.0,
        side=TradeSide.BUY,
    )
    trade_model = await repo.save_trade(trade)
    assert trade_model.id is not None
    assert trade_model.price == 150.0
