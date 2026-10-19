"""Tests for Kraken WebSocket Market Data Provider and Normalization."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from core.events import NormalizedQuoteEvent, NormalizedTradeEvent, RawMarketEvent, TradeSide
from data.ingestion.kraken import KrakenWebSocketProvider
from data.normalization.normalizer import EventNormalizer


def test_kraken_symbol_mapping():
    """Verify symbol mapping between standard tickers and Kraken-specific pairs."""
    provider = KrakenWebSocketProvider(symbols=["BTC-USD", "ETH-USD", "SOL-USD"])
    assert "XBT/USD" in provider.kraken_pairs
    assert "ETH/USD" in provider.kraken_pairs
    assert "SOL/USD" in provider.kraken_pairs


def test_kraken_parse_trade_message():
    """Verify raw Kraken trade array message is correctly parsed into RawMarketEvent."""
    provider = KrakenWebSocketProvider(symbols=["BTC-USD"])
    now = datetime.now(UTC)
    raw_kraken_msg = [
        42,
        [
            ["67250.50000", "0.25000000", "1710000000.123456", "b", "l", ""],
            ["67251.00000", "0.10000000", "1710000001.654321", "s", "m", ""],
        ],
        "trade",
        "XBT/USD",
    ]

    events = provider._parse_kraken_message(raw_kraken_msg, received_at=now)
    assert len(events) == 2

    ev1 = events[0]
    assert ev1.provider == "kraken"
    assert ev1.symbol == "BTC-USD"
    assert ev1.channel == "trade"
    assert ev1.raw_payload["price"] == "67250.50000"
    assert ev1.raw_payload["volume"] == "0.25000000"
    assert ev1.raw_payload["side"] == "b"

    ev2 = events[1]
    assert ev2.symbol == "BTC-USD"
    assert ev2.raw_payload["side"] == "s"


def test_kraken_parse_spread_message():
    """Verify raw Kraken spread array message is correctly parsed into RawMarketEvent."""
    provider = KrakenWebSocketProvider(symbols=["BTC-USD"])
    now = datetime.now(UTC)
    raw_kraken_msg = [
        101,
        ["67250.00000", "67250.50000", "1710000002.500000", "1.52300000", "2.10000000"],
        "spread",
        "XBT/USD",
    ]

    events = provider._parse_kraken_message(raw_kraken_msg, received_at=now)
    assert len(events) == 1
    ev = events[0]
    assert ev.provider == "kraken"
    assert ev.symbol == "BTC-USD"
    assert ev.channel == "spread"
    assert ev.raw_payload["bid"] == "67250.00000"
    assert ev.raw_payload["ask"] == "67250.50000"
    assert ev.raw_payload["bid_volume"] == "1.52300000"
    assert ev.raw_payload["ask_volume"] == "2.10000000"


def test_kraken_normalization_trade():
    """Verify EventNormalizer transforms Kraken trade into NormalizedTradeEvent."""
    normalizer = EventNormalizer()
    now = datetime.now(UTC)
    raw_event = RawMarketEvent(
        provider="kraken",
        symbol="BTC-USD",
        channel="trade",
        raw_payload={
            "price": "67500.25",
            "volume": "0.15",
            "time": "1710000000.0",
            "side": "b",
            "pair": "XBT/USD",
        },
        received_at=now,
    )

    trade = normalizer.normalize(raw_event)
    assert isinstance(trade, NormalizedTradeEvent)
    assert trade.symbol == "BTC-USD"
    assert trade.price == 67500.25
    assert trade.size == 0.15
    assert trade.side == TradeSide.BUY
    assert trade.provider == "kraken"


def test_kraken_normalization_spread():
    """Verify EventNormalizer transforms Kraken spread into NormalizedQuoteEvent."""
    normalizer = EventNormalizer()
    now = datetime.now(UTC)
    raw_event = RawMarketEvent(
        provider="kraken",
        symbol="ETH-USD",
        channel="spread",
        raw_payload={
            "bid": "3500.10",
            "ask": "3500.50",
            "bid_volume": "10.0",
            "ask_volume": "15.5",
            "time": "1710000000.0",
            "pair": "ETH/USD",
        },
        received_at=now,
    )

    quote = normalizer.normalize(raw_event)
    assert isinstance(quote, NormalizedQuoteEvent)
    assert quote.symbol == "ETH-USD"
    assert quote.bid_price == 3500.10
    assert quote.ask_price == 3500.50
    assert quote.bid_size == 10.0
    assert quote.ask_size == 15.5
    assert quote.provider == "kraken"


@pytest.mark.asyncio
async def test_kraken_get_historical_candles():
    """Verify Kraken public REST OHLC fetching and parsing."""
    provider = KrakenWebSocketProvider(symbols=["BTC-USD"])
    start_dt = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end_dt = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)

    mock_resp = {
        "error": [],
        "result": {
            "XXBTZUSD": [
                [int(start_dt.timestamp()), "42000.0", "42500.0", "41900.0", "42300.0", "42100.0", "15.5", 120]
            ],
            "last": int(end_dt.timestamp()),
        },
    }

    mock_resp_obj = AsyncMock()
    mock_resp_obj.raise_for_status = lambda: None
    mock_resp_obj.json = lambda: mock_resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp_obj)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    with patch("httpx.AsyncClient", return_value=mock_client):
        candles = await provider.get_historical_candles(
            symbol="BTC-USD",
            timeframe="1m",
            start_time=start_dt,
            end_time=end_dt,
        )

    assert len(candles) == 1
    c = candles[0]
    assert c.symbol == "BTC-USD"
    assert c.open == 42000.0
    assert c.high == 42500.0
    assert c.low == 41900.0
    assert c.close == 42300.0
    assert c.volume == 15.5
    assert c.is_closed is True
