"""Trading Intelligence Platform - State Snapshot & Disaster Recovery Engine.

Provides automated point-in-time state serialization, cryptographic SHA-256 checksumming,
tamper-evident archive persistence, and cold-start state rehydration for portfolio equity,
active positions, trade ledger, and risk limits.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from core.logging import get_logger
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import PaperPosition

logger = get_logger("recovery_snapshot")


class SnapshotMetadata(BaseModel):
    """Integrity and lineage metadata for a platform state snapshot."""

    snapshot_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    checksum_sha256: str
    portfolio_equity: float
    cash: float
    active_positions_count: int
    file_path: Optional[str] = None


class PlatformStatePayload(BaseModel):
    """Comprehensive snapshot of active platform state."""

    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    peak_equity: float
    positions: List[Dict[str, Any]]
    trade_count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SnapshotIntegrityError(Exception):
    """Raised when snapshot SHA-256 checksum does not match payload content."""
    pass


class SnapshotManager:
    """Manages creation, verification, and rehydration of state snapshots."""

    def __init__(self, snapshot_dir: Optional[Path] = None) -> None:
        self.snapshot_dir = snapshot_dir or Path("data/snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self, payload_dict: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 checksum over canonical JSON payload."""
        serialized = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def create_snapshot(
        self,
        engine: PaperTradingEngine,
        snapshot_id: Optional[str] = None,
    ) -> SnapshotMetadata:
        """Serialize current trading engine state and persist with SHA-256 checksum."""
        sid = snapshot_id or f"snap_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        target_file = self.snapshot_dir / f"{sid}.json"

        # Build payload
        positions_dump = [p.model_dump() for p in engine.positions.values()]
        unrealized = sum(p.unrealized_pnl for p in engine.positions.values())
        realized = sum(p.realized_pnl for p in getattr(engine, "closed_positions", []))
        peak_eq = getattr(engine, "peak_equity", engine.equity)
        trades_count = len(getattr(engine, "orders", getattr(engine, "trade_history", [])))
        payload = PlatformStatePayload(
            cash=engine.cash,
            equity=engine.equity,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            peak_equity=peak_eq,
            positions=positions_dump,
            trade_count=trades_count,
        )

        payload_dict = payload.model_dump()
        checksum = self._compute_checksum(payload_dict)

        metadata = SnapshotMetadata(
            snapshot_id=sid,
            created_at=datetime.now(timezone.utc),
            checksum_sha256=checksum,
            portfolio_equity=engine.equity,
            cash=engine.cash,
            active_positions_count=len(engine.positions),
            file_path=str(target_file.resolve()),
        )

        full_data = {
            "metadata": metadata.model_dump(),
            "payload": payload_dict,
        }

        # Write atomically
        temp_file = target_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2, default=str)
        temp_file.replace(target_file)

        logger.info(
            "State snapshot created successfully",
            snapshot_id=sid,
            equity=round(engine.equity, 2),
            positions=len(engine.positions),
            checksum=checksum[:12],
        )

        return metadata

    def restore_from_snapshot(
        self,
        snapshot_id: str,
        engine: PaperTradingEngine,
    ) -> SnapshotMetadata:
        """Validate SHA-256 checksum and rehydrate engine state from verified snapshot."""
        # Find snapshot file
        file_path = self.snapshot_dir / f"{snapshot_id}.json"
        if not file_path.exists():
            # Check if full path was passed
            alt_path = Path(snapshot_id)
            if alt_path.exists():
                file_path = alt_path
            else:
                raise FileNotFoundError(f"Snapshot file not found: {snapshot_id}")

        with open(file_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)

        meta_data = full_data.get("metadata", {})
        payload_dict = full_data.get("payload", {})

        expected_checksum = meta_data.get("checksum_sha256", "")
        recomputed_checksum = self._compute_checksum(payload_dict)

        if expected_checksum != recomputed_checksum:
            logger.error(
                "Snapshot integrity check failed",
                expected=expected_checksum[:12],
                actual=recomputed_checksum[:12],
            )
            raise SnapshotIntegrityError(
                f"Snapshot {snapshot_id} corrupted or tampered! "
                f"Expected SHA-256 {expected_checksum[:12]}, recomputed {recomputed_checksum[:12]}"
            )

        # Rehydrate PaperTradingEngine
        engine.cash = float(payload_dict["cash"])
        if hasattr(engine, "peak_equity"):
            engine.peak_equity = float(payload_dict.get("peak_equity", payload_dict["cash"]))
        if hasattr(engine, "realized_pnl"):
            engine.realized_pnl = float(payload_dict.get("realized_pnl", 0.0))

        # Rehydrate positions
        engine.positions.clear()
        for p_data in payload_dict.get("positions", []):
            pos = PaperPosition(**p_data)
            engine.positions[pos.symbol] = pos

        metadata = SnapshotMetadata(**meta_data)
        logger.info(
            "Engine state restored from snapshot",
            snapshot_id=metadata.snapshot_id,
            restored_cash=round(engine.cash, 2),
            restored_positions=len(engine.positions),
        )

        return metadata

    def list_snapshots(self) -> List[SnapshotMetadata]:
        """List all valid snapshots found in the snapshot directory sorted newest first."""
        results: List[SnapshotMetadata] = []
        for file in sorted(self.snapshot_dir.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "metadata" in data:
                    results.append(SnapshotMetadata(**data["metadata"]))
            except Exception as e:
                logger.warning("Failed to parse snapshot metadata", file=str(file), error=str(e))
        return results
