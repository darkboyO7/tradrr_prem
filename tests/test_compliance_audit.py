"""Tests for Institutional Compliance Audit Trail Engine and Cryptographic Ledger."""

from datetime import datetime, timezone
import pytest

from compliance.audit import (
    AuditLogEntry,
    AuditRecordType,
    ImmutableAuditLedger,
)


def test_immutable_audit_ledger_record_and_verify():
    """Verify cryptographic SHA-256 hash chaining and integrity verification."""
    ledger = ImmutableAuditLedger()

    # Record series of events
    e1 = ledger.record_event(
        record_type=AuditRecordType.SYSTEM_CONFIG_CHANGE,
        action="CONFIG_LOADED",
        payload={"environment": "production"},
        actor="ADMIN_AGENT",
    )
    assert e1.sequence == 0
    assert e1.previous_hash == ImmutableAuditLedger.GENESIS_HASH
    assert len(e1.entry_hash) == 64

    e2 = ledger.record_event(
        record_type=AuditRecordType.ORDER_SUBMISSION,
        action="SUBMIT_LIMIT_BUY",
        payload={"symbol": "BTC-USD", "quantity": 1.0, "price": 65000.0},
        actor="SOR_ROUTER",
    )
    assert e2.sequence == 1
    assert e2.previous_hash == e1.entry_hash

    e3 = ledger.record_event(
        record_type=AuditRecordType.ORDER_FILL,
        action="FILL_BUY",
        payload={"symbol": "BTC-USD", "executed_quantity": 1.0, "price": 65000.0},
        actor="BINANCE_CLIENT",
    )
    assert e3.sequence == 2
    assert e3.previous_hash == e2.entry_hash

    # Integrity must be valid
    is_valid, broken_seq = ledger.verify_integrity()
    assert is_valid is True
    assert broken_seq is None
    assert ledger.total_entries == 3


def test_audit_ledger_tamper_detection():
    """Verify that any modification to an existing audit record breaks cryptographic hash verification."""
    ledger = ImmutableAuditLedger()

    ledger.record_event(AuditRecordType.ORDER_SUBMISSION, "SUBMIT", {"qty": 1.0})
    ledger.record_event(AuditRecordType.ORDER_FILL, "FILL", {"qty": 1.0, "price": 65000.0})
    ledger.record_event(AuditRecordType.RISK_OVERRIDE, "OVERRIDE", {"reason": "Manager approval"})

    # Clean verification
    assert ledger.verify_integrity()[0] is True

    # Tamper with record #1 (e.g. modify price from 65,000 to 60,000)
    ledger._entries[1].payload["price"] = 60000.0

    # Verification must now fail at sequence #1
    is_valid, broken_seq = ledger.verify_integrity()
    assert is_valid is False
    assert broken_seq == 1


def test_mifid_compliance_report_export():
    """Verify MiFID II RTS 22/25 compliant report generation."""
    ledger = ImmutableAuditLedger()
    ledger.record_event(
        record_type=AuditRecordType.ORDER_SUBMISSION,
        action="NEW_ORDER",
        payload={"symbol": "ETH-USD", "qty": 10.0},
        actor="PORTFOLIO_MANAGER",
    )

    report = ledger.export_mifid_report()
    assert len(report) == 1
    rec = report[0]
    assert rec["sequence_number"] == 0
    assert "utc_timestamp" in rec
    assert rec["event_type"] == "ORDER_SUBMISSION"
    assert rec["executing_entity"] == "PORTFOLIO_MANAGER"
    assert "cryptographic_hash" in rec
    assert "event_details" in rec
