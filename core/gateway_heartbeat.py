"""Trading Intelligence Platform - Automated Self-Healing & Distributed Gateway Heartbeat Manager.

Continuous multi-venue WebSocket/REST gateway connectivity surveillance, ping RTT latency
percentile tracking (P50/P95/P99), automatic degraded connection detection, exponential
backoff self-healing reconnection, and seamless traffic failover across trading venues.
"""

from collections import deque
from datetime import datetime, timezone
from enum import Enum
import math
import random
from typing import Any, Deque, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("gateway_heartbeat")


class GatewayConnectionState(str, Enum):
    """Real-time operational state of an exchange gateway conduit."""

    CONNECTED = "CONNECTED"                               # Fully healthy, active message flow
    DEGRADED = "DEGRADED"                                 # High latency, packet jitter, or intermittent stalls
    DISCONNECTED = "DISCONNECTED"                         # Heartbeat timeout or socket drop
    RECONNECTING = "RECONNECTING"                         # Active self-healing backoff retry
    CIRCUIT_BREAKER_TRIPPED = "CIRCUIT_BREAKER_TRIPPED"   # Repeated failed retries, isolated to prevent stampede
    MAINTENANCE = "MAINTENANCE"                           # Scheduled exchange-wide maintenance window


class GatewayHealthMetrics(BaseModel):
    """Live latency, packet flow, and health metrics for a venue gateway."""

    venue: str
    connection_state: GatewayConnectionState = GatewayConnectionState.CONNECTED
    is_primary: bool = True
    failover_venue: Optional[str] = None
    last_heartbeat_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_message_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    current_ping_rtt_ms: float = 20.0
    p50_latency_ms: float = 20.0
    p95_latency_ms: float = 35.0
    p99_latency_ms: float = 50.0
    missed_heartbeats_count: int = 0
    packet_loss_rate_pct: float = 0.0
    consecutive_reconnect_attempts: int = 0
    total_reconnections_count: int = 0
    is_routing_active: bool = True
    status_message: str = "Operating nominally"


class GatewayHeartbeatReport(BaseModel):
    """Multi-venue gateway connectivity and failover posture report."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall_health: str = "ALL_SYSTEMS_NOMINAL"
    active_primary_venue: str
    gateways: Dict[str, GatewayHealthMetrics] = {}
    degraded_gateways: List[str] = []
    disconnected_gateways: List[str] = []
    failover_triggered: bool = False
    failover_reason: Optional[str] = None


class GatewayHeartbeatManager:
    """Automated self-healing coordinator for multi-exchange connectivity and gateway failover."""

    DEFAULT_VENUES = ["coinbase", "binance", "kraken"]

    def __init__(
        self,
        venues: Optional[List[str]] = None,
        primary_venue: str = "coinbase",
        heartbeat_interval_seconds: float = 5.0,
        max_missed_heartbeats: int = 3,
        degraded_p95_threshold_ms: float = 250.0,
        critical_disconnect_threshold_seconds: float = 15.0,
        max_reconnect_attempts_before_trip: int = 5,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
    ) -> None:
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.max_missed_heartbeats = max_missed_heartbeats
        self.degraded_p95_threshold_ms = degraded_p95_threshold_ms
        self.critical_disconnect_threshold_seconds = critical_disconnect_threshold_seconds
        self.max_reconnect_attempts_before_trip = max_reconnect_attempts_before_trip
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds

        venue_list = venues or self.DEFAULT_VENUES
        self.primary_venue: str = primary_venue if primary_venue in venue_list else venue_list[0]
        self.active_routing_venue: str = self.primary_venue

        self.gateways: Dict[str, GatewayHealthMetrics] = {}
        # Sliding windows for latency measurements (max 100 samples per venue)
        self._latency_samples: Dict[str, Deque[float]] = {}

        now = datetime.now(timezone.utc)
        for v in venue_list:
            is_prim = (v == self.primary_venue)
            # Assign next venue as fallback
            next_idx = (venue_list.index(v) + 1) % len(venue_list)
            fallback = venue_list[next_idx]

            self.gateways[v] = GatewayHealthMetrics(
                venue=v,
                connection_state=GatewayConnectionState.CONNECTED,
                is_primary=is_prim,
                failover_venue=fallback,
                last_heartbeat_at=now,
                last_message_at=now,
                is_routing_active=is_prim,
            )
            self._latency_samples[v] = deque(maxlen=20)
            self._latency_samples[v].extend([20.0, 22.0, 19.0, 25.0])


    def record_ping(self, venue: str, rtt_ms: float, timestamp: Optional[datetime] = None) -> None:
        """Record successful ping/pong round trip and update latency percentiles."""
        if venue not in self.gateways:
            return

        ts = timestamp or datetime.now(timezone.utc)
        gw = self.gateways[venue]
        gw.last_heartbeat_at = ts
        gw.current_ping_rtt_ms = rtt_ms
        gw.missed_heartbeats_count = 0

        # Update sliding window and percentiles
        samples = self._latency_samples[venue]
        samples.append(rtt_ms)

        arr = np.array(samples)
        gw.p50_latency_ms = round(float(np.percentile(arr, 50)), 2)
        gw.p95_latency_ms = round(float(np.percentile(arr, 95)), 2)
        gw.p99_latency_ms = round(float(np.percentile(arr, 99)), 2)

        # Check if recovered from degraded
        if gw.connection_state == GatewayConnectionState.DEGRADED and gw.p95_latency_ms < (self.degraded_p95_threshold_ms * 0.8):
            gw.connection_state = GatewayConnectionState.CONNECTED
            gw.status_message = "Latency recovered to nominal"
            logger.info(f"Gateway {venue} recovered from DEGRADED to CONNECTED (P95={gw.p95_latency_ms:.1f}ms)")
        elif gw.p95_latency_ms >= self.degraded_p95_threshold_ms:
            gw.connection_state = GatewayConnectionState.DEGRADED
            gw.status_message = f"High latency jitter: P95={gw.p95_latency_ms:.1f}ms exceeds threshold ({self.degraded_p95_threshold_ms}ms)"

    def record_message(self, venue: str, timestamp: Optional[datetime] = None) -> None:
        """Record inbound market data tick or order update message."""
        if venue in self.gateways:
            ts = timestamp or datetime.now(timezone.utc)
            self.gateways[venue].last_message_at = ts

    def calculate_reconnect_backoff_seconds(self, attempts: int) -> float:
        """Compute exponential backoff with full jitter to avoid synchronized thundering herd."""
        raw_backoff = min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** max(0, attempts - 1)))
        # Full jitter between 0.5 * backoff and 1.5 * backoff
        jittered = raw_backoff * (0.5 + random.random())
        return round(min(self.max_backoff_seconds, jittered), 2)

    def trigger_reconnect(self, venue: str) -> bool:
        """Initiate automated self-healing reconnection attempt for a disconnected gateway."""
        if venue not in self.gateways:
            return False

        gw = self.gateways[venue]
        gw.consecutive_reconnect_attempts += 1
        gw.total_reconnections_count += 1

        if gw.consecutive_reconnect_attempts > self.max_reconnect_attempts_before_trip:
            gw.connection_state = GatewayConnectionState.CIRCUIT_BREAKER_TRIPPED
            gw.status_message = f"Circuit breaker tripped after {gw.consecutive_reconnect_attempts} failed reconnects"
            logger.error(f"Gateway {venue} circuit breaker tripped. Manual intervention required.")
            return False

        gw.connection_state = GatewayConnectionState.RECONNECTING
        backoff = self.calculate_reconnect_backoff_seconds(gw.consecutive_reconnect_attempts)
        gw.status_message = f"Reconnecting attempt #{gw.consecutive_reconnect_attempts} (backoff {backoff:.1f}s)"
        logger.warning(f"Gateway {venue} scheduled reconnection: attempt {gw.consecutive_reconnect_attempts}, backoff={backoff}s")
        return True

    def acknowledge_connection_restored(self, venue: str) -> None:
        """Acknowledge successful socket handshake and restore connected operational status."""
        if venue not in self.gateways:
            return

        now = datetime.now(timezone.utc)
        gw = self.gateways[venue]
        gw.connection_state = GatewayConnectionState.CONNECTED
        gw.consecutive_reconnect_attempts = 0
        gw.missed_heartbeats_count = 0
        gw.last_heartbeat_at = now
        gw.last_message_at = now
        gw.status_message = "Connection successfully restored"
        logger.info(f"Gateway {venue} connection restored and verified healthy")

        # If primary recovered, restore routing to primary
        if gw.is_primary and self.active_routing_venue != self.primary_venue:
            logger.info(f"Primary gateway {venue} recovered. Restoring routing from {self.active_routing_venue} to primary.")
            self.gateways[self.active_routing_venue].is_routing_active = False
            gw.is_routing_active = True
            self.active_routing_venue = venue

    def evaluate_gateways(self, now: Optional[datetime] = None) -> GatewayHeartbeatReport:
        """Perform automated surveillance across all gateways, trigger failover, and synthesize status report."""
        curr_time = now or datetime.now(timezone.utc)
        degraded: List[str] = []
        disconnected: List[str] = []
        failover_triggered = False
        failover_reason = None

        for venue, gw in self.gateways.items():
            if gw.connection_state in [GatewayConnectionState.MAINTENANCE, GatewayConnectionState.CIRCUIT_BREAKER_TRIPPED]:
                if gw.connection_state == GatewayConnectionState.CIRCUIT_BREAKER_TRIPPED:
                    disconnected.append(venue)
                continue

            time_since_hb = (curr_time - gw.last_heartbeat_at).total_seconds()
            time_since_msg = (curr_time - gw.last_message_at).total_seconds()

            # Check missed heartbeats
            missed = int(time_since_hb // self.heartbeat_interval_seconds)
            gw.missed_heartbeats_count = max(0, missed)

            # Check for critical disconnect
            if time_since_hb > self.critical_disconnect_threshold_seconds:
                if gw.connection_state != GatewayConnectionState.DISCONNECTED and gw.connection_state != GatewayConnectionState.RECONNECTING:
                    gw.connection_state = GatewayConnectionState.DISCONNECTED
                    gw.status_message = f"Disconnected: No heartbeat received for {time_since_hb:.1f}s"
                    logger.warning(f"Gateway {venue} flagged DISCONNECTED (no heartbeat for {time_since_hb:.1f}s)")
                    self.trigger_reconnect(venue)
                disconnected.append(venue)

            # Check for degraded connection
            elif gw.p95_latency_ms >= self.degraded_p95_threshold_ms:
                gw.connection_state = GatewayConnectionState.DEGRADED
                degraded.append(venue)

            elif gw.missed_heartbeats_count >= 1:
                gw.connection_state = GatewayConnectionState.DEGRADED
                gw.status_message = f"Degraded: Missed {gw.missed_heartbeats_count} heartbeat(s)"
                degraded.append(venue)

        # Handle Routing Failover if currently active routing venue is disconnected, reconnecting, or tripped
        curr_active_gw = self.gateways.get(self.active_routing_venue)
        if curr_active_gw and curr_active_gw.connection_state in [
            GatewayConnectionState.DISCONNECTED,
            GatewayConnectionState.RECONNECTING,
            GatewayConnectionState.CIRCUIT_BREAKER_TRIPPED,
        ]:
            # Must failover immediately
            fallback = curr_active_gw.failover_venue or "binance"

            # Find first connected healthy alternative
            for cand_v, cand_gw in self.gateways.items():
                if cand_gw.connection_state == GatewayConnectionState.CONNECTED:
                    fallback = cand_v
                    break

            if fallback != self.active_routing_venue and self.gateways[fallback].connection_state == GatewayConnectionState.CONNECTED:
                failover_triggered = True
                failover_reason = f"Active venue {self.active_routing_venue} is {curr_active_gw.connection_state}; routing rerouted to {fallback}"
                curr_active_gw.is_routing_active = False
                self.gateways[fallback].is_routing_active = True
                self.active_routing_venue = fallback
                logger.critical(f"AUTOMATED GATEWAY FAILOVER: {failover_reason}")

        overall = "ALL_SYSTEMS_NOMINAL"
        if disconnected:
            overall = "CRITICAL_GATEWAY_OUTAGE"
        elif degraded:
            overall = "DEGRADED_GATEWAY_PERFORMANCE"

        return GatewayHeartbeatReport(
            timestamp=curr_time,
            overall_health=overall,
            active_primary_venue=self.active_routing_venue,
            gateways=self.gateways,
            degraded_gateways=degraded,
            disconnected_gateways=disconnected,
            failover_triggered=failover_triggered,
            failover_reason=failover_reason,
        )
