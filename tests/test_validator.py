"""Tests for Data Quality Engine and Validation Rules."""

from datetime import UTC, datetime

from core.events import Candle, NormalizedTradeEvent, RawMarketEvent, TradeSide
from data.validation.validator import DataValidator


def test_validate_raw_event_valid():
    validator = DataValidator()
    event = RawMarketEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        channel="matches",
        raw_payload={"type": "match", "price": "60000"},
    )
    is_valid, anomalies = validator.validate_raw_event(event)
    assert is_valid is True
    assert len(anomalies) == 0


def test_validate_raw_event_missing_symbol():
    validator = DataValidator()
    event = RawMarketEvent(
        provider="coinbase_ws",
        symbol="UNKNOWN",
        channel="matches",
        raw_payload={"type": "match"},
    )
    is_valid, anomalies = validator.validate_raw_event(event)
    assert len(anomalies) > 0
    assert any(a.check_name == "missing_symbol" for a in anomalies)


def test_validate_trade_valid():
    validator = DataValidator()
    trade = NormalizedTradeEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        trade_id="1001",
        timestamp=datetime.now(UTC),
        received_at=datetime.now(UTC),
        price=65000.0,
        size=0.5,
        side=TradeSide.BUY,
    )
    is_valid, anomalies = validator.validate_trade(trade)
    assert is_valid is True
    assert len(anomalies) == 0


def test_validate_trade_negative_price():
    validator = DataValidator()
    trade = NormalizedTradeEvent.model_construct(
        provider="coinbase_ws",
        symbol="BTC-USD",
        trade_id="1002",
        timestamp=datetime.now(UTC),
        received_at=datetime.now(UTC),
        price=-10.0,
        size=0.5,
        side=TradeSide.BUY,
    )
    is_valid, anomalies = validator.validate_trade(trade)
    assert is_valid is False
    assert any(a.check_name == "invalid_price" for a in anomalies)


def test_validate_trade_duplicate():
    validator = DataValidator()
    now = datetime.now(UTC)
    trade1 = NormalizedTradeEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        trade_id="DUP-100",
        timestamp=now,
        received_at=now,
        price=65000.0,
        size=0.5,
        side=TradeSide.BUY,
    )
    trade2 = NormalizedTradeEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        trade_id="DUP-100",
        timestamp=now,
        received_at=now,
        price=65000.0,
        size=0.5,
        side=TradeSide.BUY,
    )
    assert validator.validate_trade(trade1)[0] is True
    is_valid, anomalies = validator.validate_trade(trade2)
    assert is_valid is False
    assert any(a.check_name == "duplicate_trade" for a in anomalies)


def test_validate_trade_abnormal_price_spike():
    validator = DataValidator(max_price_spike_pct=0.10)
    now = datetime.now(UTC)
    trade1 = NormalizedTradeEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        trade_id="SPIKE-1",
        timestamp=now,
        received_at=now,
        price=100.0,
        size=1.0,
    )
    trade2 = NormalizedTradeEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        trade_id="SPIKE-2",
        timestamp=now,
        received_at=now,
        price=130.0,  # 30% jump
        size=1.0,
    )
    validator.validate_trade(trade1)
    is_valid, anomalies = validator.validate_trade(trade2)
    assert is_valid is False
    assert any(a.check_name == "abnormal_price_spike" for a in anomalies)


def test_validate_candle_valid():
    validator = DataValidator()
    candle = Candle(
        symbol="BTC-USD",
        timeframe="1m",
        timestamp=datetime.now(UTC),
        open=65000.0,
        high=65500.0,
        low=64800.0,
        close=65200.0,
        volume=10.5,
    )
    is_valid, anomalies = validator.validate_candle(candle)
    assert is_valid is True
    assert len(anomalies) == 0


def test_validate_candle_ohlc_violation():
    validator = DataValidator()
    # High is less than Open!
    candle = Candle(
        symbol="BTC-USD",
        timeframe="1m",
        timestamp=datetime.now(UTC),
        open=65000.0,
        high=64000.0,  # High < Open
        low=63000.0,
        close=63500.0,
        volume=10.0,
    )
    is_valid, anomalies = validator.validate_candle(candle)
    assert is_valid is False
    assert any("ohlc" in a.check_name for a in anomalies)
