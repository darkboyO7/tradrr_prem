"""Tests for Prometheus metrics telemetry and export formatting."""

from monitoring.metrics import (
    ANOMALIES_DETECTED_TOTAL,
    CANDLES_GENERATED_TOTAL,
    EVENT_LATENCY_SECONDS,
    MARKET_EVENTS_TOTAL,
    NORMALIZED_TRADES_TOTAL,
    ORDERS_EXECUTED_TOTAL,
    PORTFOLIO_DRAWDOWN_RATIO,
    PORTFOLIO_EQUITY_USD,
    PORTFOLIO_OPEN_POSITIONS,
    SIGNALS_GENERATED_TOTAL,
    get_latest_metrics_text,
)


def test_metrics_instrumentation_counters():
    """Verify Prometheus counters and gauges increment and render properly."""
    # Increment counters
    MARKET_EVENTS_TOTAL.labels(provider="binance", event_type="trade").inc(5)
    NORMALIZED_TRADES_TOTAL.labels(provider="binance", symbol="BTCUSDT").inc(3)
    ANOMALIES_DETECTED_TOTAL.labels(anomaly_type="price_spike").inc(1)
    CANDLES_GENERATED_TOTAL.labels(symbol="BTC-USD", timeframe="1m").inc(2)
    SIGNALS_GENERATED_TOTAL.labels(symbol="BTC-USD", action="BUY").inc(1)
    ORDERS_EXECUTED_TOTAL.labels(symbol="BTC-USD", side="BUY").inc(1)

    # Observe histogram
    EVENT_LATENCY_SECONDS.labels(provider="binance").observe(0.045)

    # Set gauges
    PORTFOLIO_EQUITY_USD.set(105420.50)
    PORTFOLIO_DRAWDOWN_RATIO.set(0.025)
    PORTFOLIO_OPEN_POSITIONS.set(2)

    # Generate text
    text = get_latest_metrics_text()
    assert "market_events_total" in text
    assert "normalized_trades_total" in text
    assert "anomalies_detected_total" in text
    assert "candles_generated_total" in text
    assert "portfolio_equity_usd" in text
    assert "105420.5" in text
    assert "portfolio_drawdown_ratio" in text
    assert "portfolio_open_positions" in text
