"""Tests for High-Throughput Stress Testing Harness and Latency Benchmarks."""

from monitoring.stress_harness import (
    PipelineThroughputBenchmark,
    SyntheticTickStormGenerator,
)


def test_synthetic_tick_storm_generation():
    gen = SyntheticTickStormGenerator(symbol="BTC-USD", base_price=65000.0, seed=123)
    ticks = gen.generate_ticks(count=2000, volatility=0.0001)

    assert len(ticks) == 2000
    first = ticks[0]
    assert first["symbol"] == "BTC-USD"
    assert first["price"] > 0
    assert first["best_bid"] < first["best_ask"]
    assert first["venue"] in ["coinbase", "binance", "kraken"]

    # Verify deterministic reproducibility
    gen2 = SyntheticTickStormGenerator(symbol="BTC-USD", base_price=65000.0, seed=123)
    ticks2 = gen2.generate_ticks(count=2000, volatility=0.0001)
    assert ticks[0]["price"] == ticks2[0]["price"]
    assert ticks[-1]["price"] == ticks2[-1]["price"]


def test_orderbook_ingestion_stress_benchmark():
    gen = SyntheticTickStormGenerator(symbol="BTC-USD", seed=42)
    ticks = gen.generate_ticks(count=3000)

    benchmark = PipelineThroughputBenchmark()
    res = benchmark.benchmark_orderbook_ingestion(ticks, min_throughput_threshold=1000.0)

    assert res.total_events == 3000
    assert res.elapsed_seconds > 0.0
    assert res.throughput_events_per_sec > 500.0
    assert res.latency_p50_us > 0.0
    assert res.latency_p99_us >= res.latency_p50_us
    assert res.benchmark_name == "orderbook_ingestion"


def test_normalizer_throughput_stress_benchmark():
    gen = SyntheticTickStormGenerator(symbol="BTC-USD", seed=99)
    ticks = gen.generate_ticks(count=3000)

    benchmark = PipelineThroughputBenchmark()
    res = benchmark.benchmark_normalizer_throughput(ticks, min_throughput_threshold=2000.0)

    assert res.total_events == 3000
    assert res.throughput_events_per_sec > 1000.0
    assert res.latency_p50_us < 5000.0  # sub-5ms median latency
    assert res.benchmark_name == "normalizer_throughput"
