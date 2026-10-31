"""Tests for Process Locking, Resolver Single Ownership, and Desktop Single-Instance Mechanism."""

from pathlib import Path
import pytest

from core.system_lock import ProcessFileLock
from learning.ledger import PredictionLedgerService


def test_process_file_lock_lifecycle(tmp_path):
    """Verify basic acquire, lock status, rejection of duplicate acquire, and release."""
    lock_file = tmp_path / "test.lock"
    lock1 = ProcessFileLock(lock_file)
    lock2 = ProcessFileLock(lock_file)

    # 1. First acquire succeeds
    assert lock1.acquire() is True
    assert lock1.is_acquired is True

    # 2. Status inspection confirms locked state
    status = ProcessFileLock.check_lock_status(lock_file)
    assert status["locked"] is True
    assert status["status"] == "LOCKED"

    # 3. Second acquire fails immediately (non-blocking)
    assert lock2.acquire() is False
    assert lock2.is_acquired is False

    # 4. First releases cleanly
    lock1.release()
    assert lock1.is_acquired is False

    # 5. Status inspection confirms unlocked
    status_after = ProcessFileLock.check_lock_status(lock_file)
    assert status_after["locked"] is False

    # 6. Second can now acquire
    assert lock2.acquire() is True
    assert lock2.is_acquired is True
    lock2.release()


def test_process_file_lock_context_manager(tmp_path):
    """Verify with ProcessFileLock(...) context manager auto-releases."""
    lock_file = tmp_path / "ctx.lock"
    with ProcessFileLock(lock_file) as acquired:
        assert acquired is True
        # Cannot acquire concurrently
        second = ProcessFileLock(lock_file)
        assert second.acquire() is False

    # After exit, lock is free
    third = ProcessFileLock(lock_file)
    assert third.acquire() is True
    third.release()


def test_prediction_outcome_resolver_single_owner(tmp_path):
    """Verify that only ONE process/thread owns the outcome resolver lifecycle."""
    sandbox_dir = tmp_path / "sandbox_ledger"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = sandbox_dir / "outcomes.jsonl"
    pending_file = sandbox_dir / "pending.jsonl"

    service1 = PredictionLedgerService(
        load_persisted=False,
        ledger_file=ledger_file,
        pending_ledger_file=pending_file,
    )
    service2 = PredictionLedgerService(
        load_persisted=False,
        ledger_file=ledger_file,
        pending_ledger_file=pending_file,
    )

    try:
        # 1. Primary service acquires resolver ownership
        success1 = service1.start_background_resolver(interval_seconds=0.2, require_lock=True)
        assert success1 is True
        assert service1._is_resolver_owner is True
        assert service1._is_running is True

        # 2. Secondary service attempts to start resolver on same ledger -> REJECTED
        success2 = service2.start_background_resolver(interval_seconds=0.2, require_lock=True)
        assert success2 is False
        assert service2._is_resolver_owner is False
        assert service2._is_running is False  # Does not spawn duplicate worker

        # 3. Primary service stops cleanly and releases lock
        service1.stop_background_resolver()
        assert service1._is_running is False
        assert service1._is_resolver_owner is False

        # 4. Secondary service can now acquire resolver ownership
        success2_retry = service2.start_background_resolver(interval_seconds=0.2, require_lock=True)
        assert success2_retry is True
        assert service2._is_resolver_owner is True
        assert service2._is_running is True
    finally:
        service1.stop_background_resolver()
        service2.stop_background_resolver()


def test_stale_lock_restart_recovery(tmp_path):
    """Verify that a stale lock file left on disk after ungraceful termination is cleanly recovered."""
    lock_file = tmp_path / "stale.lock"
    # Write a simulated dead PID to the file without holding a kernel lock
    lock_file.write_text("pid=999999\nstarted_at=2026-01-01T00:00:00Z\n", encoding="utf-8")

    status = ProcessFileLock.check_lock_status(lock_file)
    assert status["locked"] is False
    assert status["status"] == "UNLOCKED_OR_STALE"

    # New lock instance can acquire it immediately
    new_lock = ProcessFileLock(lock_file)
    assert new_lock.acquire() is True
    new_lock.release()
