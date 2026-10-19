"""Tests for High-Availability Failover Orchestrator and Split-Brain Fencing."""

from datetime import datetime, timedelta, timezone

from core.recovery.failover import (
    FailoverOrchestrator,
    HeartbeatMessage,
    NodeRole,
)


def test_heartbeat_generation():
    orch = FailoverOrchestrator(node_id="node_alpha", initial_role=NodeRole.PRIMARY, initial_epoch=1)
    hb1 = orch.generate_heartbeat()
    assert hb1.node_id == "node_alpha"
    assert hb1.role == NodeRole.PRIMARY
    assert hb1.term_epoch == 1
    assert hb1.sequence_number == 1

    hb2 = orch.generate_heartbeat()
    assert hb2.sequence_number == 2


def test_peer_heartbeat_processing():
    standby = FailoverOrchestrator(node_id="standby_node", initial_role=NodeRole.STANDBY, heartbeat_timeout_seconds=2.0)
    primary_hb = HeartbeatMessage(
        node_id="primary_node",
        role=NodeRole.PRIMARY,
        term_epoch=1,
        timestamp=datetime.now(timezone.utc),
        sequence_number=1,
    )

    standby.process_peer_heartbeat(primary_hb)
    assert standby.primary_node_id == "primary_node"
    assert standby.last_primary_heartbeat_time == primary_hb.timestamp
    assert standby.role == NodeRole.STANDBY


def test_automated_failover_promotion_on_timeout():
    standby = FailoverOrchestrator(
        node_id="standby_node",
        initial_role=NodeRole.STANDBY,
        heartbeat_timeout_seconds=2.0,
        initial_epoch=1,
    )

    t0 = datetime.now(timezone.utc)
    primary_hb = HeartbeatMessage(
        node_id="primary_node",
        role=NodeRole.PRIMARY,
        term_epoch=1,
        timestamp=t0,
        sequence_number=1,
    )
    standby.process_peer_heartbeat(primary_hb)

    # 1 second later: should not trigger failover
    t_1s = t0 + timedelta(seconds=1.0)
    promoted = standby.check_and_trigger_failover(current_time=t_1s)
    assert promoted is False
    assert standby.role == NodeRole.STANDBY

    # 3 seconds later: exceeds 2.0s timeout, should trigger failover
    t_3s = t0 + timedelta(seconds=3.0)
    promoted = standby.check_and_trigger_failover(current_time=t_3s)
    assert promoted is True
    assert standby.role == NodeRole.PRIMARY
    assert standby.term_epoch == 2
    assert standby.primary_node_id == "standby_node"


def test_split_brain_fencing_epoch_bump():
    primary_a = FailoverOrchestrator(
        node_id="node_a",
        initial_role=NodeRole.PRIMARY,
        initial_epoch=1,
    )

    # Node A is primary at epoch 1. A new message arrives from Node B claiming epoch 2 primary!
    higher_epoch_hb = HeartbeatMessage(
        node_id="node_b",
        role=NodeRole.PRIMARY,
        term_epoch=2,
        timestamp=datetime.now(timezone.utc),
        sequence_number=10,
    )

    primary_a.process_peer_heartbeat(higher_epoch_hb)

    # Node A must immediately fence itself and yield leadership by demoting to STANDBY
    assert primary_a.role == NodeRole.STANDBY
    assert primary_a.term_epoch == 2
    assert primary_a.primary_node_id == "node_b"


def test_cluster_status_telemetry():
    orch = FailoverOrchestrator(node_id="node_test", initial_role=NodeRole.PRIMARY, initial_epoch=3)
    status = orch.get_cluster_status()

    assert status["node_id"] == "node_test"
    assert status["role"] == "PRIMARY"
    assert status["term_epoch"] == 3
    assert status["primary_alive"] is True
    assert status["seconds_since_last_primary_heartbeat"] == 0.0
