"""Trading Intelligence Platform - Data Quality Engine.

Performs strict, multi-layer validation on raw exchange messages, normalized trades,
and OHLCV candles to catch anomalies, corrupted ticks, abnormal spikes, and leakage.
"""

import math
from collections import deque
from datetime import UTC, datetime

from core.events import (
    Candle,
    DataQualityAnomaly,
    DataQualitySeverity,
    NormalizedTradeEvent,
    RawMarketEvent,
)
from core.logging import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Comprehensive market data validator with stateful tracking for spikes and sequence gaps."""

    def __init__(
        self,
        max_future_seconds: float = 5.0,
        max_past_seconds: float = 86400.0,
        max_price_spike_pct: float = 0.15,  # 15% jump in a single tick
        max_seen_trades_cache: int = 50000,
    ) -> None:
        self.max_future_seconds = max_future_seconds
        self.max_past_seconds = max_past_seconds
        self.max_price_spike_pct = max_price_spike_pct
        self.max_seen_trades_cache = max_seen_trades_cache

        # Stateful memory per symbol
        self._last_prices: dict[str, float] = {}
        self._last_sequences: dict[str, int] = {}
        self._seen_trade_ids: set[str] = set()
        self._seen_trades_queue: deque[str] = deque()

        # Cumulative anomaly counter
        self.anomaly_counts: dict[str, int] = {}

    def _record_anomaly(self, anomaly: DataQualityAnomaly) -> None:
        key = f"{anomaly.symbol}:{anomaly.check_name}"
        self.anomaly_counts[key] = self.anomaly_counts.get(key, 0) + 1
        logger.warning(
            "Data Quality Anomaly Detected",
            symbol=anomaly.symbol,
            check=anomaly.check_name,
            severity=anomaly.severity.value,
            description=anomaly.description,
            metrics=anomaly.metrics,
        )

    def validate_raw_event(self, raw_event: RawMarketEvent) -> tuple[bool, list[DataQualityAnomaly]]:
        """Validate raw incoming payload structure and payload fields."""
        anomalies: list[DataQualityAnomaly] = []

        if not raw_event.symbol or raw_event.symbol == "UNKNOWN":
            anomalies.append(
                DataQualityAnomaly(
                    provider=raw_event.provider,
                    symbol="UNKNOWN",
                    check_name="missing_symbol",
                    severity=DataQualitySeverity.WARNING,
                    description="Raw event received without a valid symbol/product_id",
                )
            )

        if not isinstance(raw_event.raw_payload, dict) or not raw_event.raw_payload:
            anomalies.append(
                DataQualityAnomaly(
                    provider=raw_event.provider,
                    symbol=raw_event.symbol,
                    check_name="malformed_payload",
                    severity=DataQualitySeverity.ERROR,
                    description="Raw event payload is empty or not a valid dictionary",
                )
            )
            for a in anomalies:
                self._record_anomaly(a)
            return False, anomalies

        for a in anomalies:
            self._record_anomaly(a)
        return len(anomalies) == 0, anomalies

    def validate_trade(self, trade: NormalizedTradeEvent) -> tuple[bool, list[DataQualityAnomaly]]:
        """Validate normalized trade tick for price, volume, spikes, duplicates, and clock drift."""
        anomalies: list[DataQualityAnomaly] = []
        is_valid = True

        # 1. Non-negative & finite price
        if math.isnan(trade.price) or math.isinf(trade.price) or trade.price <= 0:
            anomalies.append(
                DataQualityAnomaly(
                    provider=trade.provider,
                    symbol=trade.symbol,
                    check_name="invalid_price",
                    severity=DataQualitySeverity.ERROR,
                    description=f"Trade price must be finite and > 0, got {trade.price}",
                    metrics={"price": trade.price, "trade_id": trade.trade_id},
                )
            )
            is_valid = False

        # 2. Non-negative & finite size
        if math.isnan(trade.size) or math.isinf(trade.size) or trade.size <= 0:
            anomalies.append(
                DataQualityAnomaly(
                    provider=trade.provider,
                    symbol=trade.symbol,
                    check_name="invalid_size",
                    severity=DataQualitySeverity.ERROR,
                    description=f"Trade size must be finite and > 0, got {trade.size}",
                    metrics={"size": trade.size, "trade_id": trade.trade_id},
                )
            )
            is_valid = False

        # 3. Timestamp sanity (future drift and staleness)
        now_utc = datetime.now(UTC)
        diff_secs = (trade.timestamp - now_utc).total_seconds()
        if diff_secs > self.max_future_seconds:
            anomalies.append(
                DataQualityAnomaly(
                    provider=trade.provider,
                    symbol=trade.symbol,
                    check_name="future_timestamp",
                    severity=DataQualitySeverity.CRITICAL,
                    description=f"Trade timestamp is {diff_secs:.2f}s in the future (clock drift)",
                    metrics={"trade_time": trade.timestamp.isoformat(), "now_utc": now_utc.isoformat()},
                )
            )
            is_valid = False
        elif (now_utc - trade.timestamp).total_seconds() > self.max_past_seconds:
            anomalies.append(
                DataQualityAnomaly(
                    provider=trade.provider,
                    symbol=trade.symbol,
                    check_name="stale_historical_timestamp",
                    severity=DataQualitySeverity.WARNING,
                    description=f"Trade timestamp is older than max allowed past window ({self.max_past_seconds}s)",
                )
            )

        # 4. Duplicate trade detection
        if trade.trade_id:
            if trade.trade_id in self._seen_trade_ids:
                anomalies.append(
                    DataQualityAnomaly(
                        provider=trade.provider,
                        symbol=trade.symbol,
                        check_name="duplicate_trade",
                        severity=DataQualitySeverity.WARNING,
                        description=f"Duplicate trade_id encountered: {trade.trade_id}",
                        metrics={"trade_id": trade.trade_id},
                    )
                )
                is_valid = False
            else:
                self._seen_trade_ids.add(trade.trade_id)
                self._seen_trades_queue.append(trade.trade_id)
                if len(self._seen_trades_queue) > self.max_seen_trades_cache:
                    old_id = self._seen_trades_queue.popleft()
                    self._seen_trade_ids.discard(old_id)

        # 5. Price spike check against previous price
        last_price = self._last_prices.get(trade.symbol)
        if last_price is not None and is_valid:
            pct_change = abs(trade.price - last_price) / last_price
            if pct_change > self.max_price_spike_pct:
                anomalies.append(
                    DataQualityAnomaly(
                        provider=trade.provider,
                        symbol=trade.symbol,
                        check_name="abnormal_price_spike",
                        severity=DataQualitySeverity.ERROR,
                        description=(
                            f"Single-tick price moved {pct_change * 100:.2f}% from {last_price} to {trade.price} "
                            f"(threshold: {self.max_price_spike_pct * 100:.1f}%)"
                        ),
                        metrics={
                            "previous_price": last_price,
                            "current_price": trade.price,
                            "pct_change": pct_change,
                        },
                    )
                )
                # We flag it, but can still accept or filter based on severity
                is_valid = False

        # 6. Sequence gap check
        if trade.sequence is not None:
            last_seq = self._last_sequences.get(trade.symbol)
            if last_seq is not None and trade.sequence > last_seq + 1:
                gap = trade.sequence - last_seq - 1
                anomalies.append(
                    DataQualityAnomaly(
                        provider=trade.provider,
                        symbol=trade.symbol,
                        check_name="sequence_gap",
                        severity=DataQualitySeverity.WARNING,
                        description=f"Exchange sequence gap of {gap} messages detected",
                        metrics={"expected": last_seq + 1, "received": trade.sequence, "gap": gap},
                    )
                )
            self._last_sequences[trade.symbol] = trade.sequence

        if is_valid:
            self._last_prices[trade.symbol] = trade.price

        for a in anomalies:
            self._record_anomaly(a)

        return is_valid, anomalies

    def validate_candle(self, candle: Candle) -> tuple[bool, list[DataQualityAnomaly]]:
        """Validate candle bar integrity, OHLC mathematical consistency, and positive volume."""
        anomalies: list[DataQualityAnomaly] = []
        is_valid = True

        # Check positive prices
        for price_field, val in [("open", candle.open), ("high", candle.high), ("low", candle.low), ("close", candle.close)]:
            if math.isnan(val) or math.isinf(val) or val <= 0:
                anomalies.append(
                    DataQualityAnomaly(
                        provider="candle_engine",
                        symbol=candle.symbol,
                        check_name="invalid_candle_price",
                        severity=DataQualitySeverity.ERROR,
                        description=f"Candle {price_field} must be finite and > 0, got {val}",
                    )
                )
                is_valid = False

        # Mathematical OHLC integrity: High must be >= Open, Close, Low. Low must be <= Open, Close, High.
        if is_valid:
            if not (candle.high >= candle.low):
                anomalies.append(
                    DataQualityAnomaly(
                        provider="candle_engine",
                        symbol=candle.symbol,
                        check_name="ohlc_high_less_than_low",
                        severity=DataQualitySeverity.CRITICAL,
                        description=f"Candle High ({candle.high}) < Low ({candle.low})",
                        metrics={"high": candle.high, "low": candle.low},
                    )
                )
                is_valid = False

            if candle.high < candle.open or candle.high < candle.close:
                anomalies.append(
                    DataQualityAnomaly(
                        provider="candle_engine",
                        symbol=candle.symbol,
                        check_name="ohlc_high_violation",
                        severity=DataQualitySeverity.CRITICAL,
                        description=f"Candle High ({candle.high}) is lower than Open ({candle.open}) or Close ({candle.close})",
                    )
                )
                is_valid = False

            if candle.low > candle.open or candle.low > candle.close:
                anomalies.append(
                    DataQualityAnomaly(
                        provider="candle_engine",
                        symbol=candle.symbol,
                        check_name="ohlc_low_violation",
                        severity=DataQualitySeverity.CRITICAL,
                        description=f"Candle Low ({candle.low}) is higher than Open ({candle.open}) or Close ({candle.close})",
                    )
                )
                is_valid = False

        # Volume must be non-negative
        if math.isnan(candle.volume) or candle.volume < 0:
            anomalies.append(
                DataQualityAnomaly(
                    provider="candle_engine",
                    symbol=candle.symbol,
                    check_name="negative_candle_volume",
                    severity=DataQualitySeverity.ERROR,
                    description=f"Candle volume cannot be negative, got {candle.volume}",
                )
            )
            is_valid = False

        for a in anomalies:
            self._record_anomaly(a)

        return is_valid, anomalies
