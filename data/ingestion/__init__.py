"""Data ingestion abstractions and exchange providers."""

from data.ingestion.base import MarketDataProvider
from data.ingestion.binance import BinanceWebSocketProvider
from data.ingestion.coinbase_ws import CoinbaseWebSocketProvider
from data.ingestion.kraken import KrakenWebSocketProvider

__all__ = [
    "MarketDataProvider",
    "CoinbaseWebSocketProvider",
    "BinanceWebSocketProvider",
    "KrakenWebSocketProvider",
]
