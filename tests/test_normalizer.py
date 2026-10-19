"""Tests for Event Normalization Engine."""

from datetime import UTC, datetime

from core.events import NormalizedQuoteEvent, NormalizedTradeEvent, RawMarketEvent, TradeSide
from data.normalization.normalizer import EventNormalizer


def test_normalize_coinbase_match(sample_coinbase_match_payload):
    normalizer = EventNormalizer()
    raw = RawMarketEvent(
        provider="coinbase_ws",
        symbol="BTC-USD",
        channel="matches",
        received_at=datetime.now(UTC),
        raw_payload=sample_coinbase_match_payload,
        sequence=sample_coinbase_match_payload["sequence"],
    )
    result = normalizer.normalize(raw)
    assert isinstance(result, NormalizedTradeEvent)
    assert result.symbol == "BTC-USD"
    assert result.trade_id == "987654321"
    assert result.price == 65432.10
    assert result.size == 0.15
    assert result.side == TradeSide.BUY
    assert result.sequence == 100200300


def test_normalize_coinbase_ticker(sample_coinbase_ticker_payload):
    normalizer = EventNormalizer()
    raw = RawMarketEvent(
        provider="coinbase_ws",
        symbol="ETH-USD",
        channel="ticker",
        received_at=datetime.now(UTC),
        raw_payload=sample_coinbase_ticker_payload,
        sequence=sample_coinbase_ticker_payload["sequence"],
    )
    result = normalizer.normalize(raw)
    assert isinstance(result, NormalizedQuoteEvent)
    assert result.symbol == "ETH-USD"
    assert result.bid_price == 3450.40
    assert result.ask_price == 3450.60
    assert result.bid_size == 5.5
    assert result.ask_size == 8.2


def test_normalize_unsupported_provider():
    normalizer = EventNormalizer()
    raw = RawMarketEvent(
        provider="unsupported_venue",
        symbol="X",
        channel="trades",
        raw_payload={"foo": "bar"},
    )
    assert normalizer.normalize(raw) is None
