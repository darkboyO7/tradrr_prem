"""Tests for Provider Watchdog, Stale Data Detection, and Auto-Recovery."""

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest

from core.events import ProviderState, RawMarketEvent
from data.ingestion.base import MarketDataProvider


class DummyTestProvider(MarketDataProvider):
    """Test stub for MarketDataProvider with configurable stale threshold."""

    def __init__(self, stale_threshold: float = 0.2):
        super().__init__(name="dummy_test", stale_threshold_seconds=stale_threshold)

    async def connect(self) -> None:
        self.set_state(ProviderState.CONNECTED)
        self.touch_event()
        self.start_watchdog()

    async def disconnect(self) -> None:
        self.stop_watchdog()
        self.set_state(ProviderState.DISCONNECTED)

    async def subscribe(self, symbols, channels) -> None:
        pass

    async def unsubscribe(self, symbols, channels) -> None:
        pass

    async def stream_raw_events(self) -> AsyncGenerator[RawMarketEvent, None]:
        yield RawMarketEvent(
            provider=self.name,
            symbol="BTC-USD",
            channel="test",
            received_at=datetime.now(UTC),
            raw_payload={"ping": "pong"},
        )

    async def get_historical_candles(self, symbol, timeframe, start_time, end_time):
        return []


@pytest.mark.asyncio
async def test_stale_stream_detection_and_recovery():
    provider = DummyTestProvider(stale_threshold=0.1)
    stale_signals = []

    def on_stale(name, elapsed):
        stale_signals.append((name, elapsed))

    provider.register_stale_callback(on_stale)
    await provider.connect()
    assert provider.state == ProviderState.CONNECTED

    # Wait longer than stale_threshold (0.1s + sleep)
    await asyncio.sleep(0.35)

    assert provider.state == ProviderState.STALE
    assert len(stale_signals) >= 1
    assert stale_signals[0][0] == "dummy_test"

    # New event arrives -> Should recover to CONNECTED
    provider.touch_event()
    assert provider.state == ProviderState.CONNECTED

    await provider.disconnect()
    assert provider.state == ProviderState.DISCONNECTED
