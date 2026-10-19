"""Tests for Chronological Candle Engine and Bar Formation."""

from datetime import UTC, datetime

from core.events import NormalizedTradeEvent
from data.candle.engine import CandleEngine, floor_to_minute


def test_floor_to_minute():
    dt = datetime(2026, 10, 4, 14, 25, 47, 891234, tzinfo=UTC)
    floored = floor_to_minute(dt)
    assert floored.second == 0
    assert floored.microsecond == 0
    assert floored.minute == 25


def test_candle_accumulation_same_minute():
    engine = CandleEngine(timeframe="1m")
    t0 = datetime(2026, 10, 4, 14, 0, 5, tzinfo=UTC)
    t1 = datetime(2026, 10, 4, 14, 0, 20, tzinfo=UTC)
    t2 = datetime(2026, 10, 4, 14, 0, 45, tzinfo=UTC)

    # Tick 1: open = 100, high = 100, low = 100, close = 100, vol = 1.0
    engine.process_trade(
        NormalizedTradeEvent(
            provider="coinbase_ws", symbol="BTC-USD", trade_id="1", timestamp=t0, received_at=t0, price=100.0, size=1.0
        )
    )
    # Tick 2: high = 105, close = 105, vol = 3.0
    engine.process_trade(
        NormalizedTradeEvent(
            provider="coinbase_ws", symbol="BTC-USD", trade_id="2", timestamp=t1, received_at=t1, price=105.0, size=2.0
        )
    )
    # Tick 3: low = 95, close = 95, vol = 4.0
    engine.process_trade(
        NormalizedTradeEvent(
            provider="coinbase_ws", symbol="BTC-USD", trade_id="3", timestamp=t2, received_at=t2, price=95.0, size=1.0
        )
    )

    partial = engine.get_current_partial_candle("BTC-USD")
    assert partial is not None
    assert partial.open == 100.0
    assert partial.high == 105.0
    assert partial.low == 95.0
    assert partial.close == 95.0
    assert partial.volume == 4.0
    assert partial.trade_count == 3
    assert partial.is_closed is False
    # VWAP = (100*1 + 105*2 + 95*1) / 4 = (100 + 210 + 95) / 4 = 405 / 4 = 101.25
    assert partial.vwap == 101.25


def test_candle_finalization_on_new_minute():
    closed_received = []

    def on_close(c):
        closed_received.append(c)

    engine = CandleEngine(timeframe="1m", on_candle_closed=on_close)
    t_min0 = datetime(2026, 10, 4, 14, 0, 10, tzinfo=UTC)
    t_min1 = datetime(2026, 10, 4, 14, 1, 5, tzinfo=UTC)

    # Ingest minute 0 trade
    engine.process_trade(
        NormalizedTradeEvent(
            provider="coinbase_ws", symbol="BTC-USD", trade_id="1", timestamp=t_min0, received_at=t_min0, price=100.0, size=1.0
        )
    )

    # Ingest minute 1 trade -> should close minute 0
    closed = engine.process_trade(
        NormalizedTradeEvent(
            provider="coinbase_ws", symbol="BTC-USD", trade_id="2", timestamp=t_min1, received_at=t_min1, price=102.0, size=0.5
        )
    )

    assert len(closed) == 1
    assert closed[0].is_closed is True
    assert closed[0].timestamp == datetime(2026, 10, 4, 14, 0, 0, tzinfo=UTC)
    assert closed[0].open == 100.0
    assert closed[0].close == 100.0
    assert len(closed_received) == 1

    # Check new active bar for minute 1
    active = engine.get_current_partial_candle("BTC-USD")
    assert active is not None
    assert active.timestamp == datetime(2026, 10, 4, 14, 1, 0, tzinfo=UTC)
    assert active.open == 102.0
