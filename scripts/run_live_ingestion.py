"""Trading Intelligence Platform - Live Market Data Ingestion Runner.

Starts the live ingestion pipeline, streams live Coinbase ticks, validates ticks,
constructs 1-minute candles, and persists records to the database.
"""

import asyncio
import signal
import sys

from core.cache import cache
from core.config import settings
from core.database import close_db, init_db
from core.logging import get_logger, setup_logging
from data.candle.engine import CandleEngine
from data.ingestion.coinbase_ws import CoinbaseWebSocketProvider
from data.ingestion.pipeline import LiveIngestionPipeline
from data.normalization.normalizer import EventNormalizer
from data.validation.validator import DataValidator

setup_logging(level=settings.LOG_LEVEL, json_format=settings.LOG_JSON)
logger = get_logger("live_runner")


async def main() -> None:
    """Run live ingestion pipeline until interrupted."""
    logger.info("Initializing Trading Intelligence Platform Live Data Ingestion...")

    # Initialize persistence and cache
    await init_db()
    await cache.initialize()

    # Instantiate provider and components
    provider = CoinbaseWebSocketProvider(
        symbols=["BTC-USD", "ETH-USD"],
        channels=["matches", "heartbeat"],
        stale_threshold_seconds=settings.STALE_DATA_THRESHOLD_SECONDS,
    )
    validator = DataValidator(max_price_spike_pct=0.15)
    normalizer = EventNormalizer()
    candle_engine = CandleEngine(timeframe="1m")

    pipeline = LiveIngestionPipeline(
        provider=provider,
        validator=validator,
        normalizer=normalizer,
        candle_engine=candle_engine,
    )

    stop_event = asyncio.Event()

    def handle_signal(*args: object) -> None:
        logger.info("Termination signal received. Shutting down...")
        stop_event.set()

    # Register signal handlers for clean Windows/POSIX termination
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: stop_event.set())

    logger.info("Starting pipeline...")
    await pipeline.start()

    # Monitoring reporter loop
    try:
        while not stop_event.is_set():
            await asyncio.sleep(5.0)
            metrics = pipeline.get_metrics()
            logger.info(
                "Live Pipeline Status",
                provider=metrics["provider"],
                state=metrics["provider_state"],
                uptime=metrics["uptime_seconds"],
                events=metrics["raw_events_received"],
                trades=metrics["trades_normalized"],
                candles=metrics["candles_generated"],
                anomalies=metrics["anomalies_detected"],
                latency_p50=metrics["latency"]["p50_ms"],
                latency_p95=metrics["latency"]["p95_ms"],
            )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Stopping pipeline and saving active candles...")
        await pipeline.stop()
        await cache.close()
        await close_db()
        logger.info("Live Ingestion Pipeline terminated cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
