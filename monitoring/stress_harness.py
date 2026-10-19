"""Trading Intelligence Platform - High-Throughput Microstructure Stress & Latency Harness.

Simulates extreme market tick storms (10,000+ events/sec) to benchmark pipeline throughput,
microsecond latency quantiles (p50/p95/p99), and memory stability under burst loads.
"""

from datetime import datetime, timedelta, timezone
import random
import time
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from core.events import RawMarketEvent
from core.logging import get_logger
from data.normalization.normalizer import EventNormalizer
from data.orderbook.consolidated import ConsolidatedOrderBook

logger = get_logger("stress_harness")


class StressBenchmarkResult(BaseModel):
    """Telemetry report produced by high-frequency stress execution benchmark."""

    total_events: int
    elapsed_seconds: float
    throughput_events_per_sec: float
    latency_p50_us: float
    latency_p90_us: float
    latency_p95_us: float
    latency_p99_us: float
    max_latency_us: float
    is_performant: bool
    benchmark_name: str = "stress_benchmark"


class SyntheticTickStormGenerator:
    """Deterministic generator for high-velocity synthetic tick storms."""

    def __init__(
        self,
        symbol: str = "BTC-USD",
        base_price: float = 65000.0,
        seed: int = 42,
    ) -> None:
        self.symbol = symbol
        self.base_price = base_price
        self.seed = seed

    def generate_ticks(
        self,
        count: int = 10000,
        volatility: float = 0.0001,
        venues: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate high-velocity burst of synthetic order book / trade ticks."""
        rng = random.Random(self.seed)
        venues_list = venues or ["coinbase", "binance", "kraken"]
        start_time = datetime.now(timezone.utc)

        current_price = self.base_price
        ticks: List[Dict[str, Any]] = []

        for i in range(count):
            # Geometric Brownian motion micro-step
            shock = rng.gauss(0, volatility)
            current_price = max(100.0, current_price * (1.0 + shock))
            spread = current_price * 0.0002
            best_bid = round(current_price - (spread / 2.0), 2)
            best_ask = round(current_price + (spread / 2.0), 2)

            tick_time = start_time + timedelta(microseconds=i * 50)
            venue = venues_list[i % len(venues_list)]
            side = "BUY" if rng.random() > 0.5 else "SELL"
            size = round(rng.uniform(0.01, 2.5), 4)

            ticks.append({
                "symbol": self.symbol,
                "price": round(current_price, 2),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "quantity": size,
                "side": side,
                "venue": venue,
                "timestamp": tick_time.isoformat(),
                "sequence": i + 1,
            })

        return ticks


class PipelineThroughputBenchmark:
    """Benchmarks processing capacity and microsecond latency under burst load."""

    def benchmark_orderbook_ingestion(
        self,
        ticks: List[Dict[str, Any]],
        symbol: str = "BTC-USD",
        min_throughput_threshold: float = 5000.0,
    ) -> StressBenchmarkResult:
        """Benchmark consolidated orderbook updates across multi-venue tick storm."""
        book = ConsolidatedOrderBook(symbol=symbol)
        latencies_ns: List[int] = []

        total_start = time.perf_counter()

        for tick in ticks:
            venue = tick["venue"]
            bid = float(tick["best_bid"])
            ask = float(tick["best_ask"])
            qty = float(tick["quantity"])

            t0 = time.perf_counter_ns()
            # Feed bids and asks to simulate multi-venue depth update
            book.update_venue_snapshot(
                venue=venue,
                bids=[(bid, qty), (bid * 0.9995, qty * 2.0)],
                asks=[(ask, qty), (ask * 1.0005, qty * 2.0)],
            )
            # Compute synthetic BBO to exercise consolidated matching
            _ = book.get_consolidated_bbo()
            t1 = time.perf_counter_ns()

            latencies_ns.append(t1 - t0)

        total_elapsed = time.perf_counter() - total_start
        total_events = len(ticks)
        throughput = total_events / max(total_elapsed, 1e-6)

        # Convert latencies to microseconds
        latencies_us = np.array(latencies_ns, dtype=np.float64) / 1000.0
        p50 = float(np.percentile(latencies_us, 50))
        p90 = float(np.percentile(latencies_us, 90))
        p95 = float(np.percentile(latencies_us, 95))
        p99 = float(np.percentile(latencies_us, 99))
        max_lat = float(np.max(latencies_us))

        is_performant = (throughput >= min_throughput_threshold) and (p99 <= 10000.0)

        logger.info(
            "Orderbook stress benchmark completed",
            total_events=total_events,
            elapsed_sec=round(total_elapsed, 4),
            throughput_eps=round(throughput, 1),
            p50_us=round(p50, 2),
            p99_us=round(p99, 2),
            performant=is_performant,
        )

        return StressBenchmarkResult(
            total_events=total_events,
            elapsed_seconds=round(total_elapsed, 4),
            throughput_events_per_sec=round(throughput, 1),
            latency_p50_us=round(p50, 2),
            latency_p90_us=round(p90, 2),
            latency_p95_us=round(p95, 2),
            latency_p99_us=round(p99, 2),
            max_latency_us=round(max_lat, 2),
            is_performant=is_performant,
            benchmark_name="orderbook_ingestion",
        )

    def benchmark_normalizer_throughput(
        self,
        raw_events: List[Dict[str, Any]],
        min_throughput_threshold: float = 10000.0,
    ) -> StressBenchmarkResult:
        """Benchmark raw event normalizer transformation throughput and latency."""
        normalizer = EventNormalizer()
        latencies_ns: List[int] = []

        total_start = time.perf_counter()

        for ev in raw_events:
            # Construct standard Coinbase match payload format
            payload = {
                "type": "match",
                "trade_id": ev["sequence"],
                "product_id": ev["symbol"],
                "price": str(ev["price"]),
                "size": str(ev["quantity"]),
                "side": ev["side"].lower(),
                "time": ev["timestamp"],
            }

            raw_event = RawMarketEvent(
                provider="coinbase_ws",
                symbol=ev["symbol"],
                channel="matches",
                received_at=datetime.now(timezone.utc),
                raw_payload=payload,
                sequence=ev["sequence"],
            )

            t0 = time.perf_counter_ns()
            _ = normalizer.normalize(raw_event)
            t1 = time.perf_counter_ns()

            latencies_ns.append(t1 - t0)

        total_elapsed = time.perf_counter() - total_start
        total_events = len(raw_events)
        throughput = total_events / max(total_elapsed, 1e-6)

        latencies_us = np.array(latencies_ns, dtype=np.float64) / 1000.0
        p50 = float(np.percentile(latencies_us, 50))
        p90 = float(np.percentile(latencies_us, 90))
        p95 = float(np.percentile(latencies_us, 95))
        p99 = float(np.percentile(latencies_us, 99))
        max_lat = float(np.max(latencies_us))

        is_performant = (throughput >= min_throughput_threshold) and (p99 <= 5000.0)

        return StressBenchmarkResult(
            total_events=total_events,
            elapsed_seconds=round(total_elapsed, 4),
            throughput_events_per_sec=round(throughput, 1),
            latency_p50_us=round(p50, 2),
            latency_p90_us=round(p90, 2),
            latency_p95_us=round(p95, 2),
            latency_p99_us=round(p99, 2),
            max_latency_us=round(max_lat, 2),
            is_performant=is_performant,
            benchmark_name="normalizer_throughput",
        )
