"""Trading Intelligence Platform - Provider-Neutral Market Data Interface.

Abstract base architecture for market data acquisition from crypto exchanges,
traditional stock exchanges (NSE/BSE), and broker feeds.
"""

import abc
import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any

from core.events import Candle, ProviderState, RawMarketEvent
from core.logging import get_logger

logger = get_logger(__name__)


class MarketDataProvider(abc.ABC):
    """Abstract interface defining required behaviors for any market data provider."""

    def __init__(self, name: str, stale_threshold_seconds: float = 15.0) -> None:
        self.name = name
        self.stale_threshold_seconds = stale_threshold_seconds
        self._state: ProviderState = ProviderState.DISCONNECTED
        self._last_event_timestamp: float | None = None
        self._stale_watchdog_task: asyncio.Task[None] | None = None
        self._on_stale_callbacks: list[Callable[[str, float], Any]] = []
        self._subscribed_symbols: set[str] = set()
        self._subscribed_channels: set[str] = set()
        self._latencies_ms: list[float] = []
        self._max_latency_samples = 1000

    @property
    def state(self) -> ProviderState:
        """Current operational state of the provider."""
        return self._state

    def set_state(self, new_state: ProviderState) -> None:
        """Transition provider state with structured audit logging."""
        prev = self._state
        self._state = new_state
        logger.info(
            "Market data provider state changed",
            provider=self.name,
            previous_state=prev.value,
            new_state=new_state.value
        )

    def register_stale_callback(self, callback: Callable[[str, float], Any]) -> None:
        """Register a hook to trigger when data stream freshness expires."""
        self._on_stale_callbacks.append(callback)

    def record_latency(self, latency_ms: float) -> None:
        """Track receive and transit latency for statistical percentile reporting."""
        if latency_ms >= 0:
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > self._max_latency_samples:
                self._latencies_ms.pop(0)

    def get_latency_metrics(self) -> dict[str, float]:
        """Compute p50, p95, and p99 latency metrics."""
        if not self._latencies_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "count": 0}

        sorted_latencies = sorted(self._latencies_ms)
        n = len(sorted_latencies)
        p50 = sorted_latencies[int(n * 0.50)]
        p95 = sorted_latencies[min(int(n * 0.95), n - 1)]
        p99 = sorted_latencies[min(int(n * 0.99), n - 1)]
        return {
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "count": n
        }

    def touch_event(self) -> None:
        """Mark event arrival for stale detection."""
        self._last_event_timestamp = time.monotonic()
        if self._state == ProviderState.STALE:
            self.set_state(ProviderState.CONNECTED)

    async def _run_stale_watchdog(self) -> None:
        """Background watchdog detecting silent stream freezes or dropped sockets."""
        check_interval = max(0.05, min(1.0, self.stale_threshold_seconds / 2.0))
        logger.info("Stale data watchdog started", provider=self.name, threshold_secs=self.stale_threshold_seconds)
        while True:
            try:
                await asyncio.sleep(check_interval)
                if self._state == ProviderState.CONNECTED and self._last_event_timestamp is not None:
                    elapsed = time.monotonic() - self._last_event_timestamp
                    if elapsed > self.stale_threshold_seconds:
                        self.set_state(ProviderState.STALE)
                        logger.warning(
                            "Market data stream is STALE",
                            provider=self.name,
                            silence_seconds=round(elapsed, 2),
                            threshold_seconds=self.stale_threshold_seconds
                        )
                        for cb in self._on_stale_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(cb):
                                    await cb(self.name, elapsed)
                                else:
                                    cb(self.name, elapsed)
                            except Exception as exc:
                                logger.error("Error executing stale callback", error=str(exc))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in stale data watchdog", error=str(exc))

    def start_watchdog(self) -> None:
        """Start the stale stream detection watchdog."""
        if self._stale_watchdog_task is None or self._stale_watchdog_task.done():
            self._stale_watchdog_task = asyncio.create_task(self._run_stale_watchdog())

    def stop_watchdog(self) -> None:
        """Stop the stale stream detection watchdog."""
        if self._stale_watchdog_task and not self._stale_watchdog_task.done():
            self._stale_watchdog_task.cancel()
            self._stale_watchdog_task = None

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the market data source."""
        pass

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Cleanly terminate connection to the market data source."""
        pass

    @abc.abstractmethod
    async def subscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Subscribe to specific symbols and channels."""
        pass

    @abc.abstractmethod
    async def unsubscribe(self, symbols: list[str], channels: list[str]) -> None:
        """Unsubscribe from specific symbols and channels."""
        pass

    @abc.abstractmethod
    def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        """Stream raw incoming market events."""
        pass

    @abc.abstractmethod
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> list[Candle]:
        """Retrieve historical candles for backfilling or calibration."""
        pass
