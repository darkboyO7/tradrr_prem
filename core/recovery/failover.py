"""Trading Intelligence Platform - High-Availability & Failover Orchestrator.

Manages primary/standby node roles, heartbeats, automatic failover detection,
and term-epoch fencing to prevent split-brain dual-primary scenarios in live trading.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("failover_orchestrator")


class NodeRole(str, Enum):
    """Execution role of a platform node in a high-availability cluster."""

    PRIMARY = "PRIMARY"      # Active leader executing orders and ingestion
    STANDBY = "STANDBY"      # Passive follower hot-monitoring state and heartbeats
    RECOVERING = "RECOVERING"  # Rehydrating state from snapshot prior to joining cluster


class HeartbeatMessage(BaseModel):
    """Cluster heartbeat broadcast."""

    node_id: str
    role: NodeRole
    term_epoch: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sequence_number: int = 0


class FailoverOrchestrator:
    """Coordinates leader election, lease-based heartbeats, and split-brain fencing."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        initial_role: NodeRole = NodeRole.STANDBY,
        heartbeat_timeout_seconds: float = 3.0,
        initial_epoch: int = 1,
    ) -> None:
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:6]}"
        self.role = initial_role
        self.term_epoch = initial_epoch
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.last_primary_heartbeat_time: Optional[datetime] = None
        self.primary_node_id: Optional[str] = None
        self._seq = 0

    def generate_heartbeat(self) -> HeartbeatMessage:
        """Create an outbound heartbeat beacon."""
        self._seq += 1
        return HeartbeatMessage(
            node_id=self.node_id,
            role=self.role,
            term_epoch=self.term_epoch,
            timestamp=datetime.now(timezone.utc),
            sequence_number=self._seq,
        )

    def process_peer_heartbeat(self, msg: HeartbeatMessage) -> None:
        """Process incoming heartbeat from a peer node with split-brain fencing."""
        # 1. Higher epoch observed: adopt it and yield leadership if necessary
        if msg.term_epoch > self.term_epoch:
            logger.warning(
                "Observed higher epoch from peer, updating term epoch",
                current_epoch=self.term_epoch,
                peer_epoch=msg.term_epoch,
                peer_node=msg.node_id,
            )
            self.term_epoch = msg.term_epoch
            if self.role == NodeRole.PRIMARY and msg.role == NodeRole.PRIMARY:
                # Fencing: Demote self to STANDBY to prevent split-brain
                self.role = NodeRole.STANDBY
                logger.error("Split-brain avoided: Demoted self to STANDBY", node_id=self.node_id)

        # 2. Track primary's liveness
        if msg.role == NodeRole.PRIMARY and msg.term_epoch >= self.term_epoch:
            self.last_primary_heartbeat_time = msg.timestamp
            self.primary_node_id = msg.node_id

            # If we were primary on equal epoch but peer has higher node_id (tie-breaker), step down
            if self.role == NodeRole.PRIMARY and msg.node_id > self.node_id:
                self.role = NodeRole.STANDBY
                logger.warning("Tie-breaker demotion to STANDBY", node_id=self.node_id)

    def check_and_trigger_failover(self, current_time: Optional[datetime] = None) -> bool:
        """Evaluate whether primary has timed out and promote standby to primary.
        
        Returns:
            True if failover promotion was triggered, False otherwise.
        """
        if self.role == NodeRole.PRIMARY:
            return False

        now = current_time or datetime.now(timezone.utc)

        # If primary has never sent a heartbeat or elapsed time exceeds timeout
        if self.last_primary_heartbeat_time is None:
            # Standby initiating initial leadership claim
            return self._promote_to_primary()

        elapsed = (now - self.last_primary_heartbeat_time).total_seconds()
        if elapsed > self.heartbeat_timeout_seconds:
            logger.critical(
                "Primary heartbeat timed out! Triggering automated failover promotion",
                elapsed_seconds=round(elapsed, 2),
                timeout_threshold=self.heartbeat_timeout_seconds,
                stale_primary=self.primary_node_id,
            )
            return self._promote_to_primary()

        return False

    def _promote_to_primary(self) -> bool:
        """Execute leader election promotion: increment epoch and claim PRIMARY role."""
        self.term_epoch += 1
        self.role = NodeRole.PRIMARY
        self.primary_node_id = self.node_id
        self.last_primary_heartbeat_time = datetime.now(timezone.utc)

        logger.info(
            "Node successfully promoted to PRIMARY",
            node_id=self.node_id,
            new_epoch=self.term_epoch,
        )
        return True

    def get_cluster_status(self) -> Dict[str, Any]:
        """Return operational telemetry on node role, epoch, and primary liveness."""
        now = datetime.now(timezone.utc)
        primary_alive = False
        time_since_hb = None

        if self.role == NodeRole.PRIMARY:
            primary_alive = True
            time_since_hb = 0.0
        elif self.last_primary_heartbeat_time:
            time_since_hb = (now - self.last_primary_heartbeat_time).total_seconds()
            primary_alive = time_since_hb <= self.heartbeat_timeout_seconds

        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "term_epoch": self.term_epoch,
            "primary_node_id": self.primary_node_id,
            "primary_alive": primary_alive,
            "seconds_since_last_primary_heartbeat": round(time_since_hb, 2) if time_since_hb is not None else None,
            "heartbeat_timeout_threshold": self.heartbeat_timeout_seconds,
        }
