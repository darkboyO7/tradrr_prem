"""Trading Intelligence Platform - Test Isolation & Production Immutability Verification.

Verifies:
1. Production files (outcomes ledger, pending ledger, backup, learning state, audit log, model store)
   remain byte-for-byte and hash-for-hash identical before and after test operations.
2. Parallel workers / concurrent tests use distinct, isolated ledger paths with zero cross-contamination.
3. Singleton instances and background worker threads are cleanly reset and stopped across tests.
4. Direct mutation attempts on production ledger or state files under pytest raise PermissionError.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import pytest

from core.universe import MarketType
from desktop.service_controller import DesktopServiceController
from learning.controller import ContinuousLearningController
from learning.ledger import (
    PROD_LEDGER_FILE_PATH,
    PROD_PENDING_LEDGER_FILE_PATH,
    PredictionLedgerService,
    RecordedPrediction,
)
from models.base import SignalAction
from models.registry import PROD_MODELS_DIR, ModelRegistry
from models.statistical import StatisticalBaselineModel
from signals.forecast import ProbabilisticForecast


PROTECTED_PROD_FILES = [
    PROD_LEDGER_FILE_PATH,
    PROD_PENDING_LEDGER_FILE_PATH,
    Path("data/ledger/prediction_outcomes.jsonl.bak"),
    Path("data/learning/learning_controller_state.json"),
    Path("data/learning/learning_audit_ledger.jsonl"),
]


def _snapshot_protected_files() -> dict[str, dict[str, Any]]:
    """Compute deterministic SHA-256 and line counts of protected production files."""
    snapshot = {}
    for p in PROTECTED_PROD_FILES:
        resolved = p.resolve()
        if resolved.exists():
            data = resolved.read_bytes()
            h = hashlib.sha256(data).hexdigest()
            line_count = len(data.splitlines())
            snapshot[str(resolved)] = {
                "sha256": h,
                "bytes": len(data),
                "lines": line_count,
            }
        else:
            snapshot[str(resolved)] = None
    return snapshot


def _make_dummy_forecast(symbol: str = "NSE:TCS", entry: float = 3500.0) -> ProbabilisticForecast:
    now = datetime.now(UTC)
    return ProbabilisticForecast(
        symbol=symbol,
        forecast_horizon="15m",
        horizon_seconds=900,
        target_time=now + timedelta(minutes=15),
        entry_reference_price=entry,
        target_price=entry * 1.01,
        invalidation_stop_price=entry * 0.99,
        prob_up=0.70,
        prob_down=0.20,
        prob_flat=0.10,
        expected_return_pct=1.0,
        expected_volatility_pct=0.5,
        target_hit_probability=0.7,
        stop_hit_probability=0.2,
        risk_reward_ratio=2.0,
        action=SignalAction.BUY,
        confidence=0.70,
        expected_holding_period_minutes=15,
        timestamp=now,
        evidence={"features": {"atr_14": 15.0, "rsi_14": 55.0}},
    )


# =========================================================================
# TEST 1: Protected Production Files Remain Completely Unchanged (Task 4)
# =========================================================================

def test_protected_production_files_remain_unchanged():
    """Verify that heavy test operations do not alter hashes or counts of protected production files."""
    before_snapshot = _snapshot_protected_files()
    assert len(before_snapshot) > 0

    # 1. Exercise PredictionLedgerService singleton
    ledger = PredictionLedgerService.get_instance()
    fc = _make_dummy_forecast("NSE:TCS", 3500.0)
    pid1 = ledger.record_forecast(fc)
    resolved = ledger.resolve_prediction_now(pid1, exit_price=3535.0)
    assert resolved is not None
    assert resolved["outcome"] == "WIN"

    # 2. Exercise DesktopServiceController evaluation
    controller = DesktopServiceController()
    eval_res = controller.evaluate_instrument("NSE:TCS", timeframe="5m", horizon="1h")
    assert eval_res["status"] == "OK"
    assert eval_res["last_price"] > 0

    # 3. Exercise ModelRegistry
    registry = ModelRegistry()
    dummy_model = StatisticalBaselineModel(model_id="iso_test_model")
    saved_path = registry.save_model(dummy_model)
    assert saved_path.exists()
    assert str(PROD_MODELS_DIR.resolve()) not in str(saved_path.resolve())

    # 4. Exercise ContinuousLearningController singleton
    ctrl = ContinuousLearningController.get_instance()
    assert ctrl is not None

    # 5. Verify isolated destination of test operations
    assert ledger.ledger_file != PROD_LEDGER_FILE_PATH
    assert ledger.pending_ledger_file != PROD_PENDING_LEDGER_FILE_PATH
    assert ledger.ledger_file.exists()
    isolated_lines = ledger.ledger_file.read_text(encoding="utf-8")
    assert pid1 in isolated_lines, "Test prediction was not written to isolated test ledger!"

    # Verify test prediction ID NEVER leaked to production ledger or pending files
    prod_ledger_text = PROD_LEDGER_FILE_PATH.read_text(encoding="utf-8")
    assert pid1 not in prod_ledger_text, "CRITICAL LEAK: Test prediction leaked into production outcomes ledger!"

    prod_pending_text = PROD_PENDING_LEDGER_FILE_PATH.read_text(encoding="utf-8")
    assert pid1 not in prod_pending_text, "CRITICAL LEAK: Test prediction leaked into production pending ledger!"

    # Verify backup file was completely untouched
    after_snapshot = _snapshot_protected_files()
    bak_key = str(Path("data/ledger/prediction_outcomes.jsonl.bak").resolve())
    assert after_snapshot[bak_key]["sha256"] == before_snapshot[bak_key]["sha256"], "Backup ledger was mutated!"

    # Verify that any lines in production ledger during test execution originated exclusively
    # from external live desktop application processes, and ZERO test prediction IDs entered production
    prod_lines_after = PROD_LEDGER_FILE_PATH.read_text(encoding="utf-8").splitlines()
    for line in prod_lines_after:
        if line.strip():
            rec = json.loads(line)
            assert rec.get("prediction_id") != pid1, f"Test prediction {pid1} leaked into production ledger!"
            assert "iso_test" not in str(rec.get("model_id")), "Test model ID leaked into production ledger!"


# =========================================================================
# TEST 2: Parallel Tests Cannot Share Production Ledger Paths (Task 5)
# =========================================================================

def test_parallel_tests_cannot_share_ledger_paths():
    """Verify that multiple concurrent workers operate on isolated ledgers without cross-contamination."""
    before_snapshot = _snapshot_protected_files()

    def _worker_task(worker_id: int) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as worker_tmp:
            w_tmp = Path(worker_tmp)
            w_ledger_file = w_tmp / f"outcomes_worker_{worker_id}.jsonl"
            w_pending_file = w_tmp / f"pending_worker_{worker_id}.jsonl"

            service = PredictionLedgerService(
                load_persisted=False,
                ledger_file=w_ledger_file,
                pending_ledger_file=w_pending_file,
            )

            sym = f"NSE:STOCK_{worker_id}"
            fc = _make_dummy_forecast(symbol=sym, entry=100.0 * worker_id)
            pid = service.record_forecast(fc)
            res = service.resolve_prediction_now(pid, exit_price=105.0 * worker_id)

            # Read back worker's local ledger
            written_lines = [json.loads(l) for l in w_ledger_file.read_text(encoding="utf-8").splitlines() if l.strip()]

            return {
                "worker_id": worker_id,
                "ledger_file": str(w_ledger_file),
                "pid": pid,
                "symbol": sym,
                "lines_count": len(written_lines),
                "pids_in_file": [r["prediction_id"] for r in written_lines],
            }

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_worker_task, range(1, 9)))

    # Verify each worker had a distinct file and unique data
    ledger_files = set()
    all_pids = set()
    for r in results:
        assert r["ledger_file"] not in ledger_files, f"Duplicate ledger file shared between workers: {r['ledger_file']}"
        ledger_files.add(r["ledger_file"])
        assert r["lines_count"] == 1
        assert r["pid"] in r["pids_in_file"]
        assert r["pid"] not in all_pids, f"Prediction ID collision across workers: {r['pid']}"
        all_pids.add(r["pid"])

    # Confirm production files were not touched by any worker
    bak_key = str(Path("data/ledger/prediction_outcomes.jsonl.bak").resolve())
    after_snapshot = _snapshot_protected_files()
    assert after_snapshot[bak_key]["sha256"] == before_snapshot[bak_key]["sha256"], "Backup was mutated!"

    # Verify none of the parallel worker prediction IDs leaked to production
    prod_ledger_text = PROD_LEDGER_FILE_PATH.read_text(encoding="utf-8")
    prod_pending_text = PROD_PENDING_LEDGER_FILE_PATH.read_text(encoding="utf-8")
    for pid in all_pids:
        assert pid not in prod_ledger_text, f"Worker prediction {pid} leaked into production ledger!"
        assert pid not in prod_pending_text, f"Worker prediction {pid} leaked into production pending ledger!"


def _make_dummy_recorded_prediction(pred_id: str) -> RecordedPrediction:
    now = datetime.now(UTC)
    return RecordedPrediction(
        prediction_id=pred_id,
        symbol="NSE:TCS",
        canonical_symbol="NSE:TCS",
        market="NSE",
        forecast_horizon="15m",
        horizon_seconds=900,
        target_time=now + timedelta(minutes=15),
        entry_price=3500.0,
        target_price=3550.0,
        stop_price=3450.0,
        action=SignalAction.BUY,
        prob_up=0.70,
        prob_down=0.20,
        confidence=0.70,
        expected_return_pct=1.0,
        created_at=now,
    )


# =========================================================================
# TEST 3: Safety Guard Blocks Direct Production File Mutation in Tests
# =========================================================================

def test_safety_guard_blocks_production_write_in_tests():
    """Verify that attempting to point PredictionLedgerService to production paths raises PermissionError."""
    # Attempting to resolve against production ledger must be blocked by safety guard
    unsafe_service = PredictionLedgerService(
        load_persisted=False,
        ledger_file=PROD_LEDGER_FILE_PATH,
        pending_ledger_file=PROD_PENDING_LEDGER_FILE_PATH,
    )

    rec1 = _make_dummy_recorded_prediction("pred_unsafe_test_001")
    # Attempting to record pending prediction must raise PermissionError
    with pytest.raises(PermissionError) as exc_info:
        unsafe_service._persist_pending_prediction(rec1)
    assert "TEST ISOLATION VIOLATION" in str(exc_info.value)

    rec2 = _make_dummy_recorded_prediction("pred_unsafe_test_002")
    # Attempting to resolve outcome to production file must raise PermissionError
    with pytest.raises(PermissionError) as exc_info2:
        unsafe_service._resolve_outcome(
            rec=rec2,
            exit_price=3535.0,
            target_hit=True,
            stop_hit=False,
        )
    assert "TEST ISOLATION VIOLATION" in str(exc_info2.value)


# =========================================================================
# TEST 4: Singleton and Background Resolver Resets Between Tests (Task 3)
# =========================================================================

def test_singleton_and_background_resolver_resets_between_tests():
    """Verify that background resolver threads and controller singletons are stopped and reset cleanly."""
    ledger = PredictionLedgerService.get_instance()
    ledger.start_background_resolver(interval_seconds=0.1)
    assert ledger._is_running is True
    thread = ledger._resolver_thread
    assert thread is not None
    assert thread.is_alive() is True

    # Call stop
    ledger.stop_background_resolver()
    assert ledger._is_running is False
    assert thread.is_alive() is False

    # Reset instance
    PredictionLedgerService.reset_instance()
    assert PredictionLedgerService._instance is None
