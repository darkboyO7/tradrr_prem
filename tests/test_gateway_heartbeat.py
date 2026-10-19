"""Tests for Automated Self-Healing & Distributed Gateway Heartbeat Manager."""

from datetime import datetime, timedelta, timezone
import pytest
from core.gateway_heartbeat import (
    GatewayConnectionState,
    GatewayHealthMetrics,
    GatewayHeartbeatManager,
    GatewayHeartbeatReport,
)


def test_gateway_initialization_and_metrics():
    """Verify default gateway initialization, primary designation, and baseline metrics."""
    mgr = GatewayHeartbeatManager(venues=["coinbase", "binance", "kraken"], primary_venue="coinbase")
    assert len(mgr.gateways) == 3
    assert mgr.active_routing_venue == "coinbase"
    assert mgr.gateways["coinbase"].is_primary is True
    assert mgr.gateways["binance"].is_primary is False
    assert mgr.gateways["coinbase"].connection_state == GatewayConnectionState.CONNECTED
    assert mgr.gateways["coinbase"].p95_latency_ms > 0.0


def test_latency_tracking_and_degradation_detection():
    """Verify latency tracking and degradation trigger when P95 breaches threshold."""
    mgr = GatewayHeartbeatManager(degraded_p95_threshold_ms=100.0)

    # Feed high latency pings to kraken
    now = datetime.now(timezone.utc)
    for _ in range(10):
        mgr.record_ping("kraken", rtt_ms=150.0, timestamp=now)

    assert mgr.gateways["kraken"].p95_latency_ms >= 100.0
    assert mgr.gateways["kraken"].connection_state == GatewayConnectionState.DEGRADED

    # Recover with low latency pings
    for _ in range(20):
        mgr.record_ping("kraken", rtt_ms=25.0, timestamp=now)

    assert mgr.gateways["kraken"].connection_state == GatewayConnectionState.CONNECTED



def test_heartbeat_timeout_and_automatic_reconnection():
    """Verify missed heartbeats and disconnection triggers auto-reconnect with exponential backoff."""
    mgr = GatewayHeartbeatManager(critical_disconnect_threshold_seconds=10.0)
    t0 = datetime.now(timezone.utc)

    # Advance time by 15 seconds without pings for binance
    t1 = t0 + timedelta(seconds=15)
    report = mgr.evaluate_gateways(now=t1)

    assert "binance" in report.disconnected_gateways
    gw = mgr.gateways["binance"]
    assert gw.connection_state in [GatewayConnectionState.DISCONNECTED, GatewayConnectionState.RECONNECTING]
    assert gw.consecutive_reconnect_attempts >= 1
    assert gw.total_reconnections_count >= 1

    # Backoff calculation
    backoff1 = mgr.calculate_reconnect_backoff_seconds(1)
    backoff3 = mgr.calculate_reconnect_backoff_seconds(3)
    assert backoff1 > 0.0
    assert backoff3 >= backoff1


def test_automated_routing_failover_and_restoration():
    """Verify automated traffic failover when primary disconnects and restoration on recovery."""
    mgr = GatewayHeartbeatManager(
        venues=["coinbase", "binance", "kraken"],
        primary_venue="coinbase",
        critical_disconnect_threshold_seconds=10.0,
    )
    t0 = datetime.now(timezone.utc)
    assert mgr.active_routing_venue == "coinbase"

    # Simulate Coinbase dying (no heartbeat for 20 seconds), while Binance stays alive
    t1 = t0 + timedelta(seconds=20)
    mgr.record_ping("binance", rtt_ms=22.0, timestamp=t1)
    mgr.record_ping("kraken", rtt_ms=25.0, timestamp=t1)

    report = mgr.evaluate_gateways(now=t1)
    assert report.failover_triggered is True
    assert mgr.active_routing_venue == "binance"
    assert mgr.gateways["binance"].is_routing_active is True
    assert mgr.gateways["coinbase"].is_routing_active is False

    # Now simulate Coinbase recovering
    mgr.acknowledge_connection_restored("coinbase")
    assert mgr.active_routing_venue == "coinbase"
    assert mgr.gateways["coinbase"].is_routing_active is True
    assert mgr.gateways["binance"].is_routing_active is False


def test_circuit_breaker_trip_on_excessive_reconnects():
    """Verify circuit breaker trips to prevent stampede when max reconnection attempts are exceeded."""
    mgr = GatewayHeartbeatManager(max_reconnect_attempts_before_trip=3)

    assert mgr.trigger_reconnect("kraken") is True   # Attempt 1
    assert mgr.trigger_reconnect("kraken") is True   # Attempt 2
    assert mgr.trigger_reconnect("kraken") is True   # Attempt 3
    # 4th attempt exceeds max 3
    assert mgr.trigger_reconnect("kraken") is False
    assert mgr.gateways["kraken"].connection_state == GatewayConnectionState.CIRCUIT_BREAKER_TRIPPED
