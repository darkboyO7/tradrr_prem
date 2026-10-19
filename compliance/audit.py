"""Trading Intelligence Platform - Institutional Compliance & Audit Trail Engine.

Maintains an immutable, append-only cryptographic audit trail with SHA-256 hash-chaining
for all order submissions, fills, cancellations, risk limit overrides, and governance decisions.
Provides tamper-detection verification and MiFID II / FINRA regulatory compliance exports.
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from core.logging import get_logger

logger = get_logger("compliance_audit")


class AuditRecordType(str, Enum):
    """Institutional classification of auditable system events."""

    ORDER_SUBMISSION = "ORDER_SUBMISSION"
    ORDER_FILL = "ORDER_FILL"
    ORDER_CANCELLATION = "ORDER_CANCELLATION"
    RISK_OVERRIDE = "RISK_OVERRIDE"
    GOVERNANCE_DECISION = "GOVERNANCE_DECISION"
    SURVEILLANCE_FLAG = "SURVEILLANCE_FLAG"
    SYSTEM_CONFIG_CHANGE = "SYSTEM_CONFIG_CHANGE"


class AuditLogEntry(BaseModel):
    """Cryptographically chained immutable audit log record."""

    sequence: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    record_type: AuditRecordType
    actor: str = "SYSTEM_ENGINE"
    action: str
    payload: Dict[str, Any]
    previous_hash: str
    entry_hash: str


class ImmutableAuditLedger:
    """Cryptographic tamper-evident append-only ledger for regulatory auditability."""

    GENESIS_HASH = "0" * 64

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = storage_path
        self._entries: List[AuditLogEntry] = []
        self._lock_state = False

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    def _compute_hash(
        self,
        sequence: int,
        timestamp_str: str,
        record_type: str,
        actor: str,
        action: str,
        payload: Dict[str, Any],
        previous_hash: str,
    ) -> str:
        """Calculate deterministic SHA-256 hash over canonical entry fields."""
        serialized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        canonical_string = f"{sequence}|{timestamp_str}|{record_type}|{actor}|{action}|{serialized_payload}|{previous_hash}"
        return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()

    def record_event(
        self,
        record_type: AuditRecordType,
        action: str,
        payload: Dict[str, Any],
        actor: str = "SYSTEM_ENGINE",
        timestamp: Optional[datetime] = None,
    ) -> AuditLogEntry:
        """Append a new tamper-evident event to the cryptographic audit chain."""
        now = timestamp or datetime.now(timezone.utc)
        iso_time = now.isoformat()
        seq = len(self._entries)

        prev_hash = self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH

        entry_hash = self._compute_hash(
            sequence=seq,
            timestamp_str=iso_time,
            record_type=record_type.value,
            actor=actor,
            action=action,
            payload=payload,
            previous_hash=prev_hash,
        )

        entry = AuditLogEntry(
            sequence=seq,
            timestamp=now,
            record_type=record_type,
            actor=actor,
            action=action,
            payload=payload,
            previous_hash=prev_hash,
            entry_hash=entry_hash,
        )

        self._entries.append(entry)

        # Optional append-only file persistence
        if self.storage_path:
            try:
                with open(self.storage_path, "a", encoding="utf-8") as f:
                    f.write(entry.model_dump_json() + "\n")
            except Exception as e:
                logger.error("Failed to append audit record to file", error=str(e))

        logger.debug(
            "Audit event recorded",
            seq=seq,
            type=record_type.value,
            action=action,
            hash_prefix=entry_hash[:8],
        )

        return entry

    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        """Verify the cryptographic integrity of the entire audit chain.
        
        Returns:
            Tuple (is_valid, broken_sequence_number_if_invalid)
        """
        for i, entry in enumerate(self._entries):
            # 1. Verify sequence order
            if entry.sequence != i:
                logger.error("Audit chain sequence mismatch", expected=i, actual=entry.sequence)
                return False, i

            # 2. Verify previous hash chaining
            expected_prev_hash = self._entries[i - 1].entry_hash if i > 0 else self.GENESIS_HASH
            if entry.previous_hash != expected_prev_hash:
                logger.error(
                    "Audit hash chaining broken",
                    seq=i,
                    expected_prev=expected_prev_hash[:8],
                    actual_prev=entry.previous_hash[:8],
                )
                return False, i

            # 3. Recalculate hash of current entry and verify no tampering
            recomputed_hash = self._compute_hash(
                sequence=entry.sequence,
                timestamp_str=entry.timestamp.isoformat(),
                record_type=entry.record_type.value,
                actor=entry.actor,
                action=entry.action,
                payload=entry.payload,
                previous_hash=entry.previous_hash,
            )

            if entry.entry_hash != recomputed_hash:
                logger.error(
                    "Audit entry content tampered",
                    seq=i,
                    recorded_hash=entry.entry_hash[:8],
                    recomputed_hash=recomputed_hash[:8],
                )
                return False, i

        return True, None

    def export_mifid_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Generate regulatory extract formatted in accordance with MiFID II RTS 22/25 standards."""
        records: List[Dict[str, Any]] = []

        for entry in self._entries:
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            records.append({
                "sequence_number": entry.sequence,
                "utc_timestamp": entry.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "event_type": entry.record_type.value,
                "executing_entity": entry.actor,
                "action": entry.action,
                "cryptographic_hash": entry.entry_hash,
                "previous_hash": entry.previous_hash,
                "event_details": entry.payload,
            })

        return records
