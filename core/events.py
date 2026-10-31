"""Trading Intelligence Platform - Core Domain Events and Data Contracts.

Defines immutable Pydantic models for raw and normalized market events,
candles, provider connection states, and quality issue descriptors.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProviderState(str, Enum):
    """Lifecycle state of an external market data provider."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STALE = "STALE"
    FAILED = "FAILED"


class DataSourceState(str, Enum):
    """Explicit state classification for market data source provenance."""
    LIVE_VERIFIED = "LIVE_VERIFIED"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    DELAYED = "DELAYED"
    CACHED = "CACHED"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


class TradeSide(str, Enum):
    """Order / Trade execution side."""
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"


class MarketEventType(str, Enum):
    """System-wide event categorizations."""
    RAW_TICK = "RAW_TICK"
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    CANDLE = "CANDLE"
    HEARTBEAT = "HEARTBEAT"
    DISCONNECT = "DISCONNECT"
    RECONNECT = "RECONNECT"
    QUALITY_ANOMALY = "QUALITY_ANOMALY"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


class RawMarketEvent(BaseModel):
    """Immutable representation of a raw message received directly from an exchange."""
    model_config = ConfigDict(frozen=True)

    provider: str = Field(description="Name of the data provider (e.g. coinbase_ws)")
    symbol: str = Field(description="Market symbol as reported by exchange")
    channel: str = Field(description="Feed channel (e.g. matches, ticker)")
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the event was received locally"
    )
    raw_payload: dict[str, Any] = Field(description="Original unparsed JSON payload")
    sequence: int | None = Field(default=None, description="Sequence number if provided")


class NormalizedTradeEvent(BaseModel):
    """Canonical, normalized trade tick representation across all venues."""
    model_config = ConfigDict(frozen=True)

    provider: str
    symbol: str
    trade_id: str
    timestamp: datetime = Field(description="Exchange timestamp in UTC")
    received_at: datetime = Field(description="Local ingest timestamp in UTC")
    price: float = Field(gt=0, description="Executed trade price")
    size: float = Field(gt=0, description="Executed trade volume / size")
    side: TradeSide = Field(default=TradeSide.UNKNOWN, description="Aggressor side")
    sequence: int | None = Field(default=None, description="Exchange sequence number")
    latency_ms: float = Field(default=0.0, description="Ingest latency in milliseconds")


class NormalizedQuoteEvent(BaseModel):
    """Canonical top-of-book quote representation."""
    model_config = ConfigDict(frozen=True)

    provider: str
    symbol: str
    timestamp: datetime
    received_at: datetime
    exchange_timestamp: datetime | None = None
    bid_price: float | None = Field(default=None, gt=0)
    bid_size: float | None = Field(default=None, ge=0)
    ask_price: float | None = Field(default=None, gt=0)
    ask_size: float | None = Field(default=None, ge=0)
    sequence: int | None = None
    latency_ms: float = 0.0


class Candle(BaseModel):
    """Canonical aggregated OHLCV candle bar."""
    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: str = Field(description="e.g. '1m', '5m', '1h', '1d'")
    timestamp: datetime = Field(description="Candle start timestamp in UTC")
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    quote_volume: float = Field(default=0.0, ge=0)
    trade_count: int = Field(default=0, ge=0)
    vwap: float | None = Field(default=None, gt=0)
    is_closed: bool = Field(default=False, description="True once the period is completed")
    source_provider: str | None = Field(default=None, description="Data provider identifier (e.g. ECB_FRANKFURTER, TRADINGVIEW)")
    source_interval: str | None = Field(default=None, description="Native source interval (e.g. '1d', '5m')")
    retrieved_at: datetime | None = Field(default=None, description="UTC timestamp when data was retrieved")


class DataQualitySeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataQualityAnomaly(BaseModel):
    """Record of a detected market data flaw or violation."""
    model_config = ConfigDict(frozen=True)

    provider: str
    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    check_name: str
    severity: DataQualitySeverity
    description: str
    metrics: dict[str, Any] = Field(default_factory=dict)
