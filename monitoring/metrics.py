"""Trading Intelligence Platform - Prometheus Metrics Instrumentation.

Exposes quantitative, operational, and financial telemetry metrics for Prometheus and Grafana.
"""

from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Global dedicated registry for platform metrics
REGISTRY = CollectorRegistry(auto_describe=True)

# 1. Ingestion & Quality Counters
MARKET_EVENTS_TOTAL = Counter(
    "market_events_total",
    "Total raw market events received from data providers",
    ["provider", "event_type"],
    registry=REGISTRY,
)

NORMALIZED_TRADES_TOTAL = Counter(
    "normalized_trades_total",
    "Total normalized trade events processed",
    ["provider", "symbol"],
    registry=REGISTRY,
)

ANOMALIES_DETECTED_TOTAL = Counter(
    "anomalies_detected_total",
    "Total market data quality anomalies caught by DataValidator",
    ["anomaly_type"],
    registry=REGISTRY,
)

CANDLES_GENERATED_TOTAL = Counter(
    "candles_generated_total",
    "Total OHLCV candles completed by CandleEngine",
    ["symbol", "timeframe"],
    registry=REGISTRY,
)

# 2. Latency Telemetry
EVENT_LATENCY_SECONDS = Histogram(
    "event_latency_seconds",
    "Transit latency from exchange to ingestion in seconds",
    ["provider"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

SIGNAL_EVALUATION_LATENCY_SECONDS = Histogram(
    "signal_evaluation_latency_seconds",
    "End-to-end signal generation and fusion latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    registry=REGISTRY,
)

# 3. Signals & Trading Decisions
SIGNALS_GENERATED_TOTAL = Counter(
    "signals_generated_total",
    "Total fused trading signals generated",
    ["symbol", "action"],
    registry=REGISTRY,
)

ORDERS_EXECUTED_TOTAL = Counter(
    "orders_executed_total",
    "Total orders executed across paper or live broker layers",
    ["symbol", "side"],
    registry=REGISTRY,
)

# 4. Risk & Portfolio Gauges
PORTFOLIO_EQUITY_USD = Gauge(
    "portfolio_equity_usd",
    "Current total portfolio equity in USD",
    registry=REGISTRY,
)

PORTFOLIO_CASH_USD = Gauge(
    "portfolio_cash_usd",
    "Current available unencumbered cash balance in USD",
    registry=REGISTRY,
)

PORTFOLIO_DRAWDOWN_RATIO = Gauge(
    "portfolio_drawdown_ratio",
    "Current portfolio drawdown ratio from peak equity (0.0 to 1.0)",
    registry=REGISTRY,
)

PORTFOLIO_OPEN_POSITIONS = Gauge(
    "portfolio_open_positions",
    "Count of currently active open positions",
    registry=REGISTRY,
)

# 5. Machine Learning & Audit Gauges
MODEL_ACCURACY_RATIO = Gauge(
    "model_accuracy_ratio",
    "Rolling prediction directional accuracy of deployed models",
    ["model_name"],
    registry=REGISTRY,
)

MODEL_WIN_RATE_RATIO = Gauge(
    "model_win_rate_ratio",
    "Rolling trade win rate of model-triggered setups",
    ["model_name"],
    registry=REGISTRY,
)

SYSTEM_HEALTH_STATUS = Gauge(
    "system_health_status",
    "Platform health status (1.0 = healthy, 0.0 = degraded/unhealthy)",
    registry=REGISTRY,
)


def get_latest_metrics_text() -> str:
    """Generate Prometheus exposition format string for scraping."""
    return generate_latest(REGISTRY).decode("utf-8")


def record_pipeline_metrics(metrics: dict[str, Any]) -> None:
    """Convenience helper to record live pipeline metrics into Prometheus instruments."""
    provider = metrics.get("provider", "unknown")
    if "raw_events_received" in metrics:
        MARKET_EVENTS_TOTAL.labels(provider=provider, event_type="all").inc(0)
    if "anomalies_detected" in metrics:
        ANOMALIES_DETECTED_TOTAL.labels(anomaly_type="general").inc(0)
