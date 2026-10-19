"""Tests for Binance WebSocket Market Data Provider and Normalization."""

from datetime import UTC, datetime

from core.events import RawMarketEvent, TradeSide
from data.ingestion.binance import BinanceWebSocketProvider
from data.normalization.normalizer import EventNormalizer


def test_binance_stream_name_construction():
    """Verify Binance stream URL parameters are correctly formatted."""
    provider = BinanceWebSocketProvider(
        symbols=["BTC/USDT", "ETH-USDT"],
        channels=["trade", "kline_1m", "bookTicker"],
    )
    streams = provider._build_stream_names()
    assert "btcusdt@trade" in streams
    assert "btcusdt@kline_1m" in streams
    assert "btcusdt@bookTicker" in streams
    assert "ethusdt@trade" in streams


def test_binance_parse_trade_message():
    """Verify raw Binance trade JSON is correctly parsed into a RawMarketEvent."""
    provider = BinanceWebSocketProvider(symbols=["BTC-USDT"])
    raw_payload = {
        "e": "trade",
        "E": 1700000000000,
        "s": "BTCUSDT",
        "t": 12345678,
        "p": "65123.50",
        "q": "0.1500",
        "b": 88,
        "a": 50,
        "T": 1700000000000,
        "m": True,  # buyer is maker -> SELL taker
    }
    now = datetime.now(UTC)
    event = provider._parse_binance_message(raw_payload, received_at=now)

    assert event is not None
    assert event.provider == "binance"
    assert event.channel == "trade"
    assert event.symbol == "BTCUSDT"
    assert event.sequence == 12345678


def test_binance_parse_kline_message():
    """Verify raw Binance kline JSON is correctly parsed."""
    provider = BinanceWebSocketProvider(symbols=["BTC-USDT"])
    raw_payload = {
        "e": "kline",
        "E": 1700000060000,
        "s": "BTCUSDT",
        "k": {
            "t": 1700000000000,
            "T": 1700000059999,
            "s": "BTCUSDT",
            "i": "1m",
            "o": "65000.00",
            "c": "65100.00",
            "h": "65150.00",
            "l": "64950.00",
            "v": "12.500",
        },
    }
    now = datetime.now(UTC)
    event = provider._parse_binance_message(raw_payload, received_at=now)

    assert event is not None
    assert event.channel == "kline"
    assert event.symbol == "BTCUSDT"
    assert event.sequence == 1700000000000


def test_binance_normalization_trade():
    """Verify EventNormalizer transforms Binance trade into NormalizedTradeEvent."""
    normalizer = EventNormalizer()
    now = datetime.now(UTC)
    raw_event = RawMarketEvent(
        provider="binance",
        symbol="BTCUSDT",
        channel="trade",
        raw_payload={
            "e": "trade",
            "E": int(now.timestamp() * 1000),
            "s": "BTCUSDT",
            "t": 987654,
            "p": "65400.00",
            "q": "0.05",
            "m": False,  # buyer is taker -> BUY
        },
        received_at=now,
    )

    trade = normalizer.normalize(raw_event)
    assert trade is not None
    assert trade.symbol == "BTCUSDT"
    assert trade.price == 65400.00
    assert trade.size == 0.05
    assert trade.side == TradeSide.BUY
    assert trade.trade_id == "987654"


def test_binance_normalization_book_ticker():
    """Verify EventNormalizer transforms Binance bookTicker into NormalizedQuoteEvent."""
    normalizer = EventNormalizer()
    now = datetime.now(UTC)
    raw_event = RawMarketEvent(
        provider="binance",
        symbol="ETHUSDT",
        channel="bookTicker",
        raw_payload={
            "u": 400900217,
            "s": "ETHUSDT",
            "b": "3450.10",
            "B": "10.5",
            "a": "3450.20",
            "A": "8.2",
        },
        received_at=now,
    )

    quote = normalizer.normalize(raw_event)
    assert quote is not None
    assert quote.symbol == "ETHUSDT"
    assert quote.bid_price == 3450.10
    assert quote.ask_price == 3450.20
    assert quote.bid_size == 10.5
    assert quote.ask_size == 8.2
