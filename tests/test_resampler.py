"""Tests for Multi-Timeframe Resampler and Streaming Aggregator."""

from datetime import UTC, datetime, timedelta

from core.events import Candle
from data.candle.multi_timeframe import MultiTimeframeAggregator
from data.candle.resampler import (
    floor_time_to_timeframe,
    parse_timeframe_to_minutes,
    resample_candles,
)


def create_1m_candles(count: int, base_time: datetime, base_price: float = 100.0) -> list[Candle]:
    """Helper creating sequential 1m candles."""
    candles = []
    price = base_price
    for i in range(count):
        ts = base_time + timedelta(minutes=i)
        c = Candle(
            symbol="BTC-USD",
            timeframe="1m",
            timestamp=ts,
            open=price,
            high=price + 5.0,
            low=price - 2.0,
            close=price + 3.0,
            volume=10.0,
            quote_volume=(price + 3.0) * 10.0,
            trade_count=15,
            vwap=price + 1.5,
            is_closed=True,
        )
        price += 3.0
        candles.append(c)
    return candles


def test_parse_timeframe():
    assert parse_timeframe_to_minutes("1m") == 1
    assert parse_timeframe_to_minutes("5m") == 5
    assert parse_timeframe_to_minutes("15m") == 15
    assert parse_timeframe_to_minutes("1h") == 60
    assert parse_timeframe_to_minutes("4h") == 240
    assert parse_timeframe_to_minutes("1d") == 1440
    assert parse_timeframe_to_minutes("1w") == 10080


def test_floor_time_to_timeframe():
    dt = datetime(2026, 10, 4, 14, 23, 45, tzinfo=UTC)
    # 5m boundary -> 14:20
    assert floor_time_to_timeframe(dt, "5m") == datetime(2026, 10, 4, 14, 20, 0, tzinfo=UTC)
    # 15m boundary -> 14:15
    assert floor_time_to_timeframe(dt, "15m") == datetime(2026, 10, 4, 14, 15, 0, tzinfo=UTC)
    # 1h boundary -> 14:00
    assert floor_time_to_timeframe(dt, "1h") == datetime(2026, 10, 4, 14, 0, 0, tzinfo=UTC)
    # 4h boundary -> 12:00
    assert floor_time_to_timeframe(dt, "4h") == datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)


def test_resample_candles_5m():
    # 15 one-minute candles starting at 12:00:00 -> should form exactly 3 five-minute candles
    base_time = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    candles_1m = create_1m_candles(15, base_time, base_price=100.0)

    resampled_5m = resample_candles(candles_1m, target_timeframe="5m")

    assert len(resampled_5m) == 3
    # First 5m bar: 12:00 -> 12:05
    c0 = resampled_5m[0]
    assert c0.timestamp == datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    assert c0.timeframe == "5m"
    assert c0.open == candles_1m[0].open
    assert c0.close == candles_1m[4].close
    assert c0.high == max(c.high for c in candles_1m[0:5])
    assert c0.low == min(c.low for c in candles_1m[0:5])
    assert c0.volume == sum(c.volume for c in candles_1m[0:5])
    assert c0.trade_count == sum(c.trade_count for c in candles_1m[0:5])


def test_resample_candles_1h():
    base_time = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    candles_1m = create_1m_candles(120, base_time, base_price=100.0)

    resampled_1h = resample_candles(candles_1m, target_timeframe="1h")

    assert len(resampled_1h) == 2
    assert resampled_1h[0].timestamp == datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    assert resampled_1h[1].timestamp == datetime(2026, 10, 4, 13, 0, 0, tzinfo=UTC)
    assert resampled_1h[0].timeframe == "1h"
    assert resampled_1h[0].open == candles_1m[0].open
    assert resampled_1h[0].close == candles_1m[59].close


def test_multi_timeframe_streaming_aggregator():
    emitted_candles = []

    def on_close(c: Candle):
        emitted_candles.append(c)

    aggregator = MultiTimeframeAggregator(timeframes=["1m", "5m", "15m"], on_candle_closed=on_close)

    base_time = datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    candles_1m = create_1m_candles(16, base_time, base_price=100.0)

    for c in candles_1m:
        closed = aggregator.process_candle(c)

    # 16 minutes processed:
    # 5m candles closed at 12:05 (boundary 12:00 closed), 12:10 (boundary 12:05 closed), 12:15 (boundary 12:10 closed) -> 3 bars
    # 15m candle closed at 12:15 (boundary 12:00 closed) -> 1 bar
    # Total closed emitted: 4
    closed_5m = [c for c in emitted_candles if c.timeframe == "5m"]
    closed_15m = [c for c in emitted_candles if c.timeframe == "15m"]

    assert len(closed_5m) == 3
    assert len(closed_15m) == 1

    # Check 15m candle properties
    c15 = closed_15m[0]
    assert c15.timestamp == datetime(2026, 10, 4, 12, 0, 0, tzinfo=UTC)
    assert c15.timeframe == "15m"
    assert c15.open == candles_1m[0].open
    assert c15.close == candles_1m[14].close

    # Context query
    ctx = aggregator.get_context("BTC-USD")
    assert ctx["1m"] is not None
    assert ctx["5m"] is not None
    assert ctx["15m"] is not None
