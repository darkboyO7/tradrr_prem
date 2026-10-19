"""Tests for Health Checks and Observability Endpoints."""

import pytest

from monitoring.health import check_system_health


@pytest.mark.asyncio
async def test_check_system_health():
    dummy_pipeline_metrics = {
        "provider": "coinbase_ws",
        "provider_state": "CONNECTED",
        "uptime_seconds": 12.5,
        "raw_events_received": 100,
        "trades_normalized": 95,
        "candles_generated": 2,
        "anomalies_detected": 0,
        "latency": {"p50_ms": 12.4, "p95_ms": 25.1, "p99_ms": 40.0, "count": 95},
    }

    report = await check_system_health(pipeline_metrics=dummy_pipeline_metrics)
    assert "status" in report
    assert "uptime_seconds" in report
    assert "components" in report
    assert report["components"]["pipeline"]["provider"] == "coinbase_ws"
    assert report["components"]["pipeline"]["trades_normalized"] == 95
