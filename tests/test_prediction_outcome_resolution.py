"""Trading Intelligence Platform - Prediction Outcome Resolution Test Suite.

Comprehensive tests for:
1. A prediction that has not reached its horizon.
2. A due prediction with valid outcome data.
3. Missing outcome data (no fabricated prices, distinct expired-without-data state).
4. Resolver restart, crash recovery, and pending persistence.
5. Duplicate resolution attempts and idempotency.
6. Incorrect symbols, timestamps, and invalid prices.
7. Persistence and strict training eligibility filtering.
8. Scheduler / background resolver startup, autonomous execution, and clean shutdown.
"""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import tempfile
import time

import pytest

from core.universe import MarketType
from data.market_state import MarketStateService
from learning.controller import ContinuousLearningController
from learning.ledger import PredictionLedgerService, RecordedPrediction
from models.base import SignalAction
from signals.forecast import ProbabilisticForecast


@pytest.fixture
def temp_ledger_env():
    """Create isolated temporary files for ledger outcomes and pending predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        ledger_file = tmp_path / "prediction_outcomes.jsonl"
        pending_file = tmp_path / "pending_predictions.jsonl"
        state_file = tmp_path / "learning_controller_state.json"
        audit_file = tmp_path / "learning_audit.jsonl"

        PredictionLedgerService.reset_instance()
        ContinuousLearningController.reset_instance()

        service = PredictionLedgerService(
            load_persisted=False,
            ledger_file=ledger_file,
            pending_ledger_file=pending_file,
        )
        PredictionLedgerService._instance = service

        yield {
            "tmp_path": tmp_path,
            "service": service,
            "ledger_file": ledger_file,
            "pending_file": pending_file,
            "state_file": state_file,
            "audit_file": audit_file,
        }

        service.stop_background_resolver()
        PredictionLedgerService.reset_instance()
        ContinuousLearningController.reset_instance()


def make_test_forecast(
    symbol: str = "NSE:TCS",
    canonical_symbol: str | None = None,
    entry: float = 3500.0,
    target: float | None = None,
    stop: float | None = None,
    action: SignalAction = SignalAction.BUY,
    prob_up: float = 0.70,
    prob_down: float = 0.20,
    confidence: float = 0.70,
    horizon: str = "15m",
    horizon_seconds: int = 900,
    target_time: datetime | None = None,
    features: dict | None = None,
) -> ProbabilisticForecast:
    now = datetime.now(UTC)
    tt = target_time or (now + timedelta(seconds=horizon_seconds))
    t_price = target if target is not None else (entry * 1.01 if action == SignalAction.BUY else entry * 0.99)
    s_price = stop if stop is not None else (entry * 0.99 if action == SignalAction.BUY else entry * 1.01)
    return ProbabilisticForecast(
        symbol=symbol,
        forecast_horizon=horizon,
        horizon_seconds=horizon_seconds,
        target_time=tt,
        entry_reference_price=entry,
        target_price=t_price,
        invalidation_stop_price=s_price,
        prob_up=prob_up,
        prob_down=prob_down,
        prob_flat=max(0.0, 1.0 - (prob_up + prob_down)),
        expected_return_pct=1.0,
        expected_volatility_pct=0.5,
        target_hit_probability=0.7,
        stop_hit_probability=0.2,
        risk_reward_ratio=2.0,
        action=action,
        confidence=confidence,
        expected_holding_period_minutes=15,
        timestamp=now,
        evidence={"features": features or {"atr_14": 15.0, "rsi_14": 55.0}},
    )


# =========================================================================
# TEST 1: Prediction that has not reached its horizon
# =========================================================================

def test_prediction_not_reached_horizon(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    now = datetime.now(UTC)
    target_time = now + timedelta(minutes=15)

    fc = make_test_forecast(entry=3500.0, target=3550.0, stop=3450.0, target_time=target_time)
    pred_id = service.record_forecast(fc)

    assert pred_id in service._active_predictions
    assert service._active_predictions[pred_id].status == "PENDING"
    assert service.total_resolved == 0

    # Advance time only 5 minutes (within 15m horizon), price did not hit target or stop
    mid_time = now + timedelta(minutes=5)
    resolved = service.resolve_pending_forecasts(
        symbol_or_canonical="NSE:TCS",
        current_price=3510.0,
        current_time=mid_time,
    )

    assert len(resolved) == 0, "Premature resolution: prediction resolved before reaching horizon!"
    assert pred_id in service._active_predictions
    assert service.total_resolved == 0
    assert not temp_ledger_env["ledger_file"].exists() or temp_ledger_env["ledger_file"].stat().st_size == 0


# =========================================================================
# TEST 2: Due prediction with valid outcome data
# =========================================================================

def test_due_prediction_with_valid_outcome_data(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    now = datetime.now(UTC)
    target_time = now + timedelta(minutes=15)

    fc = make_test_forecast(entry=3500.0, target=3535.0, stop=3480.0, action=SignalAction.BUY, target_time=target_time)
    pred_id = service.record_forecast(fc, features_used={"atr_14": 15.0, "rsi": 60.0})

    # Advance time to 16 minutes (past horizon)
    exit_time = now + timedelta(minutes=16)
    exit_price = 3530.0  # Above entry -> WIN for BUY

    resolved = service.resolve_pending_forecasts(
        symbol_or_canonical="NSE:TCS",
        current_price=exit_price,
        current_time=exit_time,
    )

    assert len(resolved) == 1
    rec = resolved[0]
    assert rec["prediction_id"] == pred_id
    assert rec["status"] == "RESOLVED"
    assert rec["outcome"] == "WIN"
    assert rec["exit_price"] == exit_price
    assert rec["market_target_direction"] == "BUY"
    assert rec["market_class_int"] == 2
    assert rec["multiclass_brier_score"] is not None
    assert rec["resolved_at"] is not None
    assert pred_id not in service._active_predictions
    assert service.total_resolved == 1

    # Verify JSONL persistence
    lines = temp_ledger_env["ledger_file"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["status"] == "RESOLVED"
    assert persisted["outcome"] == "WIN"
    assert persisted["prediction_id"] == pred_id


# =========================================================================
# TEST 3: Missing outcome data (no fake prices, graceful expiration)
# =========================================================================

def test_missing_outcome_data_not_fabricated(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    now = datetime.now(UTC)
    target_time = now + timedelta(minutes=15)

    fc = make_test_forecast(symbol="NSE:UNKNOWN", entry=100.0, target_time=target_time)
    pred_id = service.record_forecast(fc)

    # 1. Past horizon (16m), but market data is completely unavailable (price <= 0.0)
    exit_time = now + timedelta(minutes=16)
    resolved = service.resolve_pending_forecasts(
        symbol_or_canonical="NSE:UNKNOWN",
        current_price=0.0,  # Missing/invalid price
        current_time=exit_time,
    )

    # Must NOT fabricate exit = entry price or resolve with fake 0 return!
    assert len(resolved) == 0, "Fabricated resolution occurred when market price was missing!"
    assert pred_id in service._active_predictions
    assert service.total_resolved == 0

    # 2. Advance past 24-hour grace period with NO market data ever arriving
    expired_time = target_time + timedelta(hours=25)
    resolved_expired = service.resolve_pending_forecasts(
        symbol_or_canonical="NSE:UNKNOWN",
        current_price=0.0,
        current_time=expired_time,
    )

    assert len(resolved_expired) == 1
    exp_rec = resolved_expired[0]
    assert exp_rec["prediction_id"] == pred_id
    assert exp_rec["status"] == "EXPIRED_WITHOUT_DATA"
    assert exp_rec["outcome"] == "NO_DATA"
    assert exp_rec["exit_price"] is None
    assert pred_id not in service._active_predictions


# =========================================================================
# TEST 4: Resolver restart and crash recovery
# =========================================================================

def test_resolver_restart_and_recovery(temp_ledger_env):
    ledger_file = temp_ledger_env["ledger_file"]
    pending_file = temp_ledger_env["pending_file"]
    service_1: PredictionLedgerService = temp_ledger_env["service"]

    now = datetime.now(UTC)
    fc = make_test_forecast(entry=3500.0, target_time=now + timedelta(minutes=15))
    pred_id = service_1.record_forecast(fc, features_used={"atr": 10.0})

    # Verify pending prediction was safely persisted to disk
    assert pending_file.exists()
    pending_lines = pending_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(pending_lines) == 1
    assert json.loads(pending_lines[0])["prediction_id"] == pred_id

    # Simulate abrupt application restart: drop service_1 from memory
    service_1.stop_background_resolver()
    del service_1

    # Instantiate service_2 with load_persisted=True
    service_2 = PredictionLedgerService(
        load_persisted=True,
        ledger_file=ledger_file,
        pending_ledger_file=pending_file,
    )

    # Active predictions must be recovered cleanly from disk
    assert pred_id in service_2._active_predictions
    assert service_2._active_predictions[pred_id].entry_price == 3500.0
    assert service_2.total_resolved == 0

    # Resolve with service_2 after time advances
    exit_time = now + timedelta(minutes=20)
    resolved = service_2.resolve_pending_forecasts(
        symbol_or_canonical="NSE:TCS",
        current_price=3540.0,
        current_time=exit_time,
    )

    assert len(resolved) == 1
    assert resolved[0]["status"] == "RESOLVED"
    assert resolved[0]["outcome"] == "WIN"
    assert pred_id not in service_2._active_predictions

    # Pending storage must now be empty
    pending_lines_after = [l for l in pending_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(pending_lines_after) == 0

    service_2.stop_background_resolver()


# =========================================================================
# TEST 5: Duplicate resolution attempts and idempotency
# =========================================================================

def test_duplicate_resolution_attempts_idempotent(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    fc = make_test_forecast(entry=2500.0, target=2535.0, stop=2480.0, action=SignalAction.BUY)
    pred_id = service.record_forecast(fc, features_used={"atr": 12.0})

    # First immediate resolution
    res1 = service.resolve_prediction_now(pred_id, exit_price=2525.0)
    assert res1 is not None
    assert res1["status"] == "RESOLVED"
    assert service.total_resolved == 1

    # Second immediate resolution (duplicate attempt)
    res2 = service.resolve_prediction_now(pred_id, exit_price=2525.0)
    assert res2 is None, "Duplicate resolution attempt should return None!"

    # Third attempt via resolve_pending_forecasts
    res3 = service.resolve_pending_forecasts(
        symbol_or_canonical="NSE:TCS",
        current_price=2525.0,
        current_time=datetime.now(UTC) + timedelta(hours=1),
    )
    assert len(res3) == 0

    # Verify JSONL contains exactly ONE record
    lines = temp_ledger_env["ledger_file"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert service.total_resolved == 1


# =========================================================================
# TEST 6: Incorrect symbols, timestamps, and invalid prices
# =========================================================================

def test_incorrect_symbols_or_timestamps(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    fc = make_test_forecast(symbol="NSE:TCS", canonical_symbol="NSE:TCS", entry=3500.0)
    pred_id = service.record_forecast(fc)

    # 1. Resolve for a different symbol -> should NOT resolve TCS
    res_wrong_sym = service.resolve_pending_forecasts(
        symbol_or_canonical="NSE:INFY",
        current_price=1600.0,
        current_time=datetime.now(UTC) + timedelta(minutes=20),
    )
    assert len(res_wrong_sym) == 0
    assert pred_id in service._active_predictions

    # 2. Call resolve_prediction_now with negative or zero price -> returns None safely
    res_neg = service.resolve_prediction_now(pred_id, exit_price=-50.0)
    assert res_neg is None
    res_zero = service.resolve_prediction_now(pred_id, exit_price=0.0)
    assert res_zero is None
    assert pred_id in service._active_predictions

    # 3. Call with nonexistent prediction ID -> returns None safely
    assert service.resolve_prediction_now("nonexistent_pid", exit_price=3500.0) is None


# =========================================================================
# TEST 7: Persistence and training eligibility filtering
# =========================================================================

def test_persistence_and_training_eligibility(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    ledger_file = temp_ledger_env["ledger_file"]

    # 1. Valid resolved prediction with features
    fc1 = make_test_forecast(symbol="NSE:TCS", entry=3500.0, target=3535.0, action=SignalAction.BUY)
    pid1 = service.record_forecast(fc1, features_used={"rsi": 65.0, "atr": 15.0})
    service.resolve_prediction_now(pid1, exit_price=3530.0)

    # 2. Prediction resolved without features (empty dict)
    fc2 = make_test_forecast(symbol="NSE:INFY", entry=1500.0, target=1530.0, action=SignalAction.BUY)
    pid2 = service.record_forecast(fc2, features_used={})
    service.resolve_prediction_now(pid2, exit_price=1520.0)

    # 3. Prediction that expired without market data
    fc3 = make_test_forecast(symbol="NSE:MISSING", entry=100.0, target_time=datetime.now(UTC) - timedelta(hours=30))
    pid3 = service.record_forecast(fc3, features_used={"rsi": 50.0})
    service.resolve_pending_forecasts(current_price=0.0, current_time=datetime.now(UTC))

    # 4. Pending prediction (not yet resolved)
    fc4 = make_test_forecast(
        symbol="NSE:RELIANCE",
        entry=2500.0,
        target=999999.0,
        stop=1.0,
        target_time=datetime.now(UTC) + timedelta(hours=5),
    )
    service.record_forecast(fc4, features_used={"rsi": 45.0})

    # Query ledger eligible learning samples
    eligible_ledger = service.get_eligible_learning_samples()
    pids = [r["prediction_id"] for r in eligible_ledger]
    assert pid1 in pids
    assert pid2 not in pids, "Record with empty features must be excluded from training!"
    assert pid3 not in pids, "EXPIRED_WITHOUT_DATA must be excluded from training!"
    assert fc4.symbol not in [r["symbol"] for r in eligible_ledger], "Pending prediction must be excluded from training!"
    assert len(eligible_ledger) == 1

    # Query controller collection
    ctrl = ContinuousLearningController(
        state_file=temp_ledger_env["state_file"],
        audit_log_file=temp_ledger_env["audit_file"],
        ledger_file=ledger_file,
    )
    ctrl_samples = ctrl._collect_eligible_outcomes()
    ctrl_pids = [r["prediction_id"] for r in ctrl_samples]
    assert ctrl_pids == [pid1]


# =========================================================================
# TEST 8: Scheduler startup, autonomous resolution, and clean shutdown
# =========================================================================

def test_scheduler_startup_and_failure_reporting(temp_ledger_env):
    service: PredictionLedgerService = temp_ledger_env["service"]
    market_state = MarketStateService.get_instance()

    # Pre-populate market price for NSE:TCS
    st = market_state.ensure_instrument_state("NSE:TCS")
    st.last_price = 3520.0

    # Start background resolver with very fast check interval (0.05s)
    service.start_background_resolver(interval_seconds=0.05)
    assert service._is_running is True
    assert service._resolver_thread is not None and service._resolver_thread.is_alive()

    # Record a forecast whose horizon is already reached (target_time in the past)
    past_time = datetime.now(UTC) - timedelta(seconds=1)
    fc = make_test_forecast(entry=3500.0, target=3535.0, stop=3480.0, action=SignalAction.BUY, target_time=past_time)
    pred_id = service.record_forecast(fc, features_used={"rsi": 55.0})

    # Wait for the background worker to resolve it autonomously
    deadline = time.time() + 3.0
    while pred_id in service._active_predictions and time.time() < deadline:
        time.sleep(0.05)

    assert pred_id not in service._active_predictions, "Background resolver failed to resolve prediction autonomously!"
    assert service.total_resolved == 1

    # Clean shutdown
    service.stop_background_resolver()
    assert service._is_running is False
    assert not service._resolver_thread.is_alive()
