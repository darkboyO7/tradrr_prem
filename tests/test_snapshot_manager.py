"""Tests for State Snapshot and Disaster Recovery Engine."""

import json
from pathlib import Path
import pytest

from core.recovery.snapshot import (
    PlatformStatePayload,
    SnapshotIntegrityError,
    SnapshotManager,
    SnapshotMetadata,
)
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, PaperPosition


def test_snapshot_creation_and_checksum(tmp_path: Path):
    manager = SnapshotManager(snapshot_dir=tmp_path)
    engine = PaperTradingEngine(initial_cash=100_000.0)

    # Simulate some active positions and PnL
    engine.cash = 85_000.0
    engine.positions["BTC-USD"] = PaperPosition(
        position_id="pos_btc_01",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        quantity=0.25,
        entry_price=60_000.0,
        current_price=62_000.0,
        stop_loss=58_000.0,
        take_profit=66_000.0,
        unrealized_pnl=500.0,
        realized_pnl=0.0,
    )

    meta = manager.create_snapshot(engine=engine, snapshot_id="test_snap_001")

    assert meta.snapshot_id == "test_snap_001"
    assert meta.portfolio_equity == engine.equity
    assert meta.cash == 85_000.0
    assert meta.active_positions_count == 1
    assert len(meta.checksum_sha256) == 64

    # Verify file content
    snap_file = tmp_path / "test_snap_001.json"
    assert snap_file.exists()

    with open(snap_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "payload" in data
    assert data["payload"]["cash"] == 85_000.0
    assert len(data["payload"]["positions"]) == 1
    assert data["payload"]["positions"][0]["symbol"] == "BTC-USD"


def test_snapshot_restore_rehydration(tmp_path: Path):
    manager = SnapshotManager(snapshot_dir=tmp_path)
    engine = PaperTradingEngine(initial_cash=50_000.0)

    engine.cash = 40_000.0
    engine.positions["ETH-USD"] = PaperPosition(
        position_id="pos_eth_01",
        symbol="ETH-USD",
        side=OrderSide.BUY,
        quantity=3.0,
        entry_price=3_000.0,
        current_price=3_100.0,
        stop_loss=2_850.0,
        take_profit=3_300.0,
        unrealized_pnl=300.0,
        realized_pnl=0.0,
    )

    manager.create_snapshot(engine=engine, snapshot_id="rehydrate_snap")

    # Create fresh empty engine
    cold_engine = PaperTradingEngine(initial_cash=100_000.0)
    assert cold_engine.cash == 100_000.0
    assert len(cold_engine.positions) == 0

    # Rehydrate
    restored_meta = manager.restore_from_snapshot("rehydrate_snap", engine=cold_engine)
    assert restored_meta.snapshot_id == "rehydrate_snap"
    assert cold_engine.cash == 40_000.0
    assert "ETH-USD" in cold_engine.positions
    assert cold_engine.positions["ETH-USD"].quantity == 3.0
    assert cold_engine.positions["ETH-USD"].entry_price == 3_000.0
    assert cold_engine.positions["ETH-USD"].stop_loss == 2_850.0


def test_snapshot_tamper_detection(tmp_path: Path):
    manager = SnapshotManager(snapshot_dir=tmp_path)
    engine = PaperTradingEngine(initial_cash=75_000.0)
    manager.create_snapshot(engine=engine, snapshot_id="tamper_snap")

    # Manually tamper with the snapshot file payload
    snap_file = tmp_path / "tamper_snap.json"
    with open(snap_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Tamper cash amount
    data["payload"]["cash"] = 999_999.0

    with open(snap_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Attempt restoration on tampered file -> must raise SnapshotIntegrityError
    cold_engine = PaperTradingEngine(initial_cash=75_000.0)
    with pytest.raises(SnapshotIntegrityError) as exc_info:
        manager.restore_from_snapshot("tamper_snap", engine=cold_engine)

    assert "corrupted or tampered" in str(exc_info.value)


def test_snapshot_list_and_missing(tmp_path: Path):
    manager = SnapshotManager(snapshot_dir=tmp_path)
    engine = PaperTradingEngine(initial_cash=100_000.0)

    manager.create_snapshot(engine=engine, snapshot_id="snap_a")
    manager.create_snapshot(engine=engine, snapshot_id="snap_b")

    snaps = manager.list_snapshots()
    assert len(snaps) == 2
    ids = [s.snapshot_id for s in snaps]
    assert "snap_a" in ids
    assert "snap_b" in ids

    # Missing snapshot restore
    with pytest.raises(FileNotFoundError):
        manager.restore_from_snapshot("non_existent_id", engine=engine)
