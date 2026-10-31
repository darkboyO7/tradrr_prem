"""Trading Intelligence Platform - Comprehensive Continuous Learning & 99% Target Acceptance Test Suite.

Verifies all 13 core requirements:
1. Complete prediction metadata persistence.
2. In-flight pending state until horizon expiration.
3. Zero-lookahead outcome resolution (MFE/MAE/Brier/Triple-Barrier).
4. Outcome deduplication and idempotent dataset ingestion.
5. Chronological dataset preparation, purging, and embargo.
6. Candidate model training, artifact creation, and SHA-256 lineage.
7. Untouched out-of-sample champion vs challenger validation.
8. Governance rejection of inferior candidate.
9. Governance promotion of superior candidate.
10. Live production inference immediately using newly promoted champion.
11. Automated production rollback on model degradation.
12. Crash recovery and persistent state machine resumption.
13. Honest statistical evaluation of 99% target with Wilson score confidence intervals.
"""

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

from learning.controller import (
    ContinuousLearningController,
    LearningMetricsReport,
    LearningState,
    Target99Status,
)
from learning.ledger import PredictionLedgerService, RecordedPrediction
from models.base import ModelPrediction, SignalAction
from models.ml_classifier import LightGBMPredictor
from models.registry import ModelRegistry
from signals.forecast import ProbabilisticForecast


@pytest.fixture
def temp_learning_env():
    """Create isolated temporary directory for state, registry, and ledger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        state_file = tmp_path / "learning_controller_state.json"
        audit_file = tmp_path / "learning_audit_ledger.jsonl"
        models_dir = tmp_path / "models_store"
        models_dir.mkdir(parents=True, exist_ok=True)

        registry = ModelRegistry(base_dir=models_dir)
        ContinuousLearningController.reset_instance()
        ledger_file = tmp_path / "prediction_outcomes.jsonl"

        ctrl = ContinuousLearningController(
            state_file=state_file,
            audit_log_file=audit_file,
            model_registry=registry,
            ledger_file=ledger_file,
            min_samples_to_retrain=10,
            min_new_outcomes_delta=2,
            min_eval_samples_for_99=30,
            rollback_win_rate_threshold=40.0,
            rollback_brier_threshold=0.35,
            rollback_window_size=10,
        )

        pending_file = tmp_path / "pending_predictions.jsonl"
        yield {
            "tmp_path": tmp_path,
            "controller": ctrl,
            "state_file": state_file,
            "audit_file": audit_file,
            "registry": registry,
            "ledger_file": ledger_file,
            "pending_file": pending_file,
        }

        ctrl.stop_background_loop()
        ContinuousLearningController.reset_instance()


def make_forecast(
    symbol: str = "NSE:TCS",
    entry: float = 3500.0,
    target: float = 3535.0,
    stop: float = 3480.0,
    action: SignalAction = SignalAction.BUY,
    prob_up: float = 0.70,
    prob_down: float = 0.20,
    confidence: float = 0.70,
    horizon: str = "15m",
    horizon_seconds: int = 900,
    target_time: datetime | None = None,
    evidence: dict | None = None,
) -> ProbabilisticForecast:
    now = datetime.now(UTC)
    tt = target_time or (now + timedelta(seconds=horizon_seconds))
    return ProbabilisticForecast(
        symbol=symbol,
        forecast_horizon=horizon,
        horizon_seconds=horizon_seconds,
        target_time=tt,
        entry_reference_price=entry,
        target_price=target,
        invalidation_stop_price=stop,
        action=action,
        prob_up=prob_up,
        prob_down=prob_down,
        prob_flat=round(max(0.0, 1.0 - prob_up - prob_down), 4),
        expected_return_pct=1.0,
        expected_volatility_pct=0.8,
        target_hit_probability=prob_up,
        stop_hit_probability=prob_down,
        risk_reward_ratio=2.0,
        confidence=confidence,
        expected_holding_period_minutes=horizon_seconds / 60.0,
        evidence=evidence or {},
    )


def test_01_prediction_recording_with_full_metadata():
    """Requirement 1: Verify prediction is recorded with all canonical metadata fields."""
    ledger = PredictionLedgerService(load_persisted=False)
    now = datetime.now(UTC)
    target_time = now + timedelta(minutes=15)

    fc = make_forecast(
        symbol="NSE:TCS",
        entry=3500.0,
        target=3535.0,
        stop=3480.0,
        action=SignalAction.BUY,
        prob_up=0.72,
        prob_down=0.18,
        confidence=0.72,
        target_time=target_time,
        evidence={"regime": "BULL_TREND", "source_timestamp": now.isoformat()},
    )

    pred_id = ledger.record_forecast(
        fc,
        model_id="lgbm_champion_v1.0",
        market="NSE",
        canonical_symbol="NSE:TCS",
        features_used={"return_1": 0.002, "rsi_14": 58.5, "macd_hist": 0.001},
        source_data_timestamp=now.isoformat(),
        feature_pipeline_version="v1.2",
        decision_thresholds={"min_confidence": 0.52},
    )

    assert pred_id in ledger._active_predictions
    rec = ledger._active_predictions[pred_id]

    assert rec.canonical_symbol == "NSE:TCS"
    assert rec.market == "NSE"
    assert rec.forecast_horizon == "15m"
    assert rec.horizon_seconds == 900
    assert rec.entry_price == 3500.0
    assert rec.target_price == 3535.0
    assert rec.stop_price == 3480.0
    assert rec.action == SignalAction.BUY
    assert rec.confidence == 0.72
    assert rec.model_id == "lgbm_champion_v1.0"
    assert rec.feature_pipeline_version == "v1.2"
    assert rec.decision_thresholds["min_confidence"] == 0.52
    assert rec.features_used["rsi_14"] == 58.5
    assert rec.resolved is False
    assert rec.data_quality_status == "PASSED"


def test_02_prediction_pending_until_horizon_or_barrier():
    """Requirement 2: Prediction remains pending until target hit, stop hit, or horizon expiration."""
    ledger = PredictionLedgerService(load_persisted=False)
    now = datetime.now(UTC)
    fc = make_forecast(
        symbol="NSE:RELIANCE",
        entry=2500.0,
        target=2550.0,
        stop=2475.0,
        action=SignalAction.BUY,
        prob_up=0.68,
        prob_down=0.20,
        confidence=0.68,
        target_time=now + timedelta(minutes=15),
    )
    pred_id = ledger.record_forecast(fc, model_id="lgbm_champion_v1.0", market="NSE")

    # Intermediate price: within barriers, time not expired -> remains pending
    resolved = ledger.resolve_pending_forecasts(
        symbol_or_canonical="NSE:RELIANCE",
        current_price=2510.0,
        current_time=now + timedelta(minutes=5),
    )
    assert len(resolved) == 0
    assert pred_id in ledger._active_predictions
    assert ledger._active_predictions[pred_id].resolved is False
    assert ledger._active_predictions[pred_id].highest_seen == 2510.0


def test_03_prediction_outcome_resolution_and_triple_barrier(temp_learning_env):
    """Requirement 3: Resolve predictions correctly against price action (Target Hit = WIN, Stop Hit = LOSS)."""
    ledger = PredictionLedgerService(
        load_persisted=False,
        ledger_file=temp_learning_env["ledger_file"],
        pending_ledger_file=temp_learning_env["pending_file"],
    )
    now = datetime.now(UTC)
    fc = make_forecast(
        symbol="NSE:INFY",
        entry=1500.0,
        target=1530.0,
        stop=1485.0,
        action=SignalAction.BUY,
        prob_up=0.70,
        prob_down=0.20,
        confidence=0.70,
        target_time=now + timedelta(minutes=15),
    )
    pred_id = ledger.record_forecast(fc, model_id="lgbm_champion_v1.0", market="NSE")

    # Upper target breached -> resolves immediately as WIN
    resolved = ledger.resolve_pending_forecasts(
        symbol_or_canonical="NSE:INFY",
        current_price=1535.0,
        current_time=now + timedelta(minutes=7),
    )
    assert len(resolved) == 1
    res = resolved[0]
    assert res["prediction_id"] == pred_id
    assert res["outcome"] == "WIN"
    assert res["target_hit"] is True
    assert res["actual_return_pct"] > 0
    assert res["mfe_pct"] >= 2.0
    # Objective market direction and multiclass Brier
    assert res["market_target_direction"] == "BUY"
    assert res["market_class_int"] == 2
    assert res["market_friction_bps"] == 6.0
    # Multiclass Brier: (0.20 - 0)^2 + (0.10 - 0)^2 + (0.70 - 1.0)^2 = 0.04 + 0.01 + 0.09 = 0.14
    assert res["multiclass_brier_score"] == pytest.approx(0.14, abs=1e-2)
    assert res["binary_brier_score"] == pytest.approx((0.70 - 1.0) ** 2, rel=1e-3)
    assert pred_id not in ledger._active_predictions


def test_04_outcome_deduplication_and_idempotency(temp_learning_env):
    """Requirement 4: Ensure outcomes are resolved exactly once without duplication."""
    ledger = PredictionLedgerService(
        load_persisted=False,
        ledger_file=temp_learning_env["ledger_file"],
        pending_ledger_file=temp_learning_env["pending_file"],
    )
    now = datetime.now(UTC)
    fc = make_forecast(
        symbol="NSE:SBIN",
        entry=800.0,
        target=815.0,
        stop=790.0,
        action=SignalAction.BUY,
        prob_up=0.65,
        prob_down=0.25,
        confidence=0.65,
        target_time=now + timedelta(minutes=15),
    )
    pred_id = ledger.record_forecast(fc)

    # First resolution
    out1 = ledger.resolve_prediction_now(pred_id, exit_price=820.0)
    assert out1 is not None
    assert out1["outcome"] == "WIN"
    assert ledger.total_resolved == 1

    # Second resolution call on same ID must be a safe no-op
    out2 = ledger.resolve_prediction_now(pred_id, exit_price=820.0)
    assert out2 is None
    assert ledger.total_resolved == 1

    # Eligible learning samples must have exactly 1 record
    eligible = ledger.get_eligible_learning_samples()
    pids = [r["prediction_id"] for r in eligible]
    assert pids.count(pred_id) == 1


def test_05_chronological_splits_and_embargo(temp_learning_env):
    """Requirement 5: Dataset preparation maintains strict chronological ordering with embargo buffer."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]

    # Synthesize chronological samples
    records = []
    base_t = datetime(2026, 10, 1, 9, 15, tzinfo=UTC)
    for i in range(25):
        t = base_t + timedelta(minutes=15 * i)
        records.append(
            {
                "prediction_id": f"pred_{i}",
                "predicted_at": t.isoformat(),
                "action": "BUY" if i % 2 == 0 else "SELL",
                "outcome": "WIN" if (i % 3 != 0) else "LOSS",
                "actual_return_pct": 0.5 if (i % 3 != 0) else -0.5,
                "confidence": 0.65,
                "features_used": {
                    "return_1": 0.001 * (i + 1),
                    "rsi_14": 40 + i,
                    "macd_hist": 0.0005 * i,
                },
            }
        )

    df_X, y_arr, ret_arr, fingerprint = ctrl._build_dataset_dataframe(records)
    assert len(df_X) == 25
    assert len(fingerprint) == 16

    # Verify chronological splits
    X_train, y_train, X_oos, y_oos, ret_oos = ctrl._prepare_chronological_splits(
        df_X, y_arr, ret_arr, train_ratio=0.65, embargo_samples=2
    )

    train_len = int(25 * 0.65)  # 16
    assert len(X_train) == train_len
    # Embargo gap of 2 samples: OOS starts at train_len + 2 = 18
    assert len(X_oos) == 25 - (train_len + 2)  # 7
    # Zero-overlap check: train and OOS index ranges must be strictly disjoint
    assert set(X_train.index).isdisjoint(set(X_oos.index))
    assert max(X_train.index) < min(X_oos.index)


def test_06_candidate_training_and_artifact_lineage(temp_learning_env):
    """Requirement 6: Train candidate model and persist artifact with cryptographic lineage."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]
    reg: ModelRegistry = temp_learning_env["registry"]

    X = pd.DataFrame(
        {
            "return_1": [0.005, -0.005, 0.002, -0.003, 0.006, -0.004, 0.001, -0.001, 0.007, -0.008] * 3,
            "rsi_14": [65, 35, 52, 45, 68, 32, 50, 48, 72, 28] * 3,
            "macd_hist": [0.001, -0.001, 0.0002, -0.0004, 0.0015, -0.0012, 0.0, -0.0001, 0.002, -0.002] * 3,
        }
    )
    y = np.array([2, 0, 1, 0, 2, 0, 1, 1, 2, 0] * 3)

    version = "v20261019_120000"
    candidate = ctrl._train_candidate(X, y, version=version)

    assert candidate.is_fitted is True
    assert candidate.metadata.model_id == f"lgbm_candidate_{version}"
    assert candidate.metadata.version == version

    # Save to registry and verify artifact files + hash
    saved_path = reg.save_model(candidate)
    assert saved_path.exists()
    assert (saved_path / "model.joblib").exists()
    assert (saved_path / "metadata.json").exists()
    assert candidate.metadata.sha256_hash is not None
    assert len(candidate.metadata.sha256_hash) == 64


def test_07_untouched_oos_champion_challenger_validation(temp_learning_env):
    """Requirement 7: Champion and challenger evaluated on identical untouched out-of-sample split."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]

    # Train two predictors
    X_train = pd.DataFrame(
        {
            "return_1": [0.005, -0.005, 0.002, -0.003, 0.006, -0.004, 0.001, -0.002, 0.007, -0.006] * 4,
            "rsi_14": [65, 35, 55, 45, 70, 30, 50, 48, 75, 25] * 4,
            "macd_hist": [0.001, -0.001, 0.0, -0.0005, 0.002, -0.002, 0.0, -0.0002, 0.002, -0.002] * 4,
        }
    )
    y_train = np.array([2, 0, 1, 0, 2, 0, 1, 0, 2, 0] * 4)

    champion = LightGBMPredictor(model_id="champ", version="v1.0", n_estimators=20).fit(X_train, y_train)
    challenger = LightGBMPredictor(model_id="chall", version="v1.1", n_estimators=40).fit(X_train, y_train)

    # Identical untouched OOS test split
    X_oos = pd.DataFrame(
        {
            "return_1": [0.004, -0.005, 0.003, -0.004, 0.006, -0.003],
            "rsi_14": [62, 38, 54, 42, 69, 36],
            "macd_hist": [0.0008, -0.0009, 0.0001, -0.0004, 0.0012, -0.0008],
        }
    )
    y_oos = np.array([2, 0, 2, 0, 2, 0])
    ret_oos = np.array([0.004, -0.005, 0.003, -0.004, 0.006, -0.003])

    champ_metrics = ctrl._evaluate_model_on_oos(champion, X_oos, y_oos, ret_oos)
    chall_metrics = ctrl._evaluate_model_on_oos(challenger, X_oos, y_oos, ret_oos)

    assert champ_metrics["oos_samples"] == len(X_oos)
    assert chall_metrics["oos_samples"] == len(X_oos)
    assert "accuracy_pct" in champ_metrics
    assert "brier_score" in champ_metrics
    assert "expectancy_pct" in champ_metrics


def test_08_governance_rejection_of_inferior_candidate(temp_learning_env):
    """Requirement 8: Candidate with degraded Brier score or single-class collapse is rejected."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]

    champ_metrics = {
        "oos_samples": 10,
        "accuracy_pct": 65.0,
        "brier_score": 0.20,
        "expectancy_pct": 0.50,
        "unique_classes_predicted": 2,
    }

    # Case A: Degraded Brier score
    cand_inferior = {
        "oos_samples": 10,
        "accuracy_pct": 50.0,
        "brier_score": 0.38,  # worse than champion
        "expectancy_pct": -0.20,
        "unique_classes_predicted": 2,
    }
    promoted, reason = ctrl._evaluate_promotion_gates(cand_inferior, champ_metrics)
    assert promoted is False
    assert "Brier score degraded" in reason or "Expectancy degraded" in reason

    # Case B: Single class collapse (degenerate predictor)
    cand_collapsed = {
        "oos_samples": 10,
        "accuracy_pct": 70.0,
        "brier_score": 0.18,
        "expectancy_pct": 0.60,
        "unique_classes_predicted": 1,  # Only predicted BUY
    }
    promoted, reason = ctrl._evaluate_promotion_gates(cand_collapsed, champ_metrics)
    assert promoted is False
    assert "collapsed to a single class" in reason


def test_09_governance_promotion_and_lineage_update(temp_learning_env):
    """Requirement 9: Candidate beating champion is promoted and lineage updated."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]
    reg: ModelRegistry = temp_learning_env["registry"]

    initial_champ = ctrl.active_champion_id

    # Create synthetic superior outcomes
    records = []
    base_t = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)
    for i in range(35):
        t = base_t + timedelta(minutes=15 * i)
        is_buy = i % 2 == 0
        records.append(
            {
                "prediction_id": f"p_{i}",
                "predicted_at": t.isoformat(),
                "action": "BUY" if is_buy else "SELL",
                "outcome": "WIN",
                "actual_return_pct": 0.8,
                "net_trade_return_pct": 0.74,
                "market_return_pct": 0.8 if is_buy else -0.8,
                "market_target_direction": "BUY" if is_buy else "SELL",
                "market_class_int": 2 if is_buy else 0,
                "entry_price": 100.0,
                "exit_price": 100.8 if is_buy else 99.2,
                "prob_up": 0.80 if is_buy else 0.10,
                "prob_down": 0.10 if is_buy else 0.80,
                "confidence": 0.80,
                "features_used": {
                    "return_1": 0.005 if is_buy else -0.005,
                    "rsi_14": 65 if is_buy else 35,
                    "macd_hist": 0.001 if is_buy else -0.001,
                    "candle_range_pct": 0.008,
                    "candle_body_ratio": 0.7,
                    "adx_14": 28.0,
                    "atr_14": 1.2,
                    "rvol_20": 1.3,
                    "volatility_regime": 1.0,
                },
            }
        )

    # Ingest into ledger (isolated for this test)
    ledger = PredictionLedgerService.get_instance()
    ledger._resolved_outcomes = list(records)

    # Force a full autonomous learning cycle
    res = ctrl.run_cycle_now(force=True)

    assert res.triggered is True
    assert res.state == LearningState.IDLE
    assert res.promoted is True
    assert ctrl.active_champion_id != initial_champ
    assert ctrl.previous_champion_id == initial_champ
    assert len(ctrl.champion_lineage) >= 1
    assert ctrl.champion_lineage[-1]["model_id"] == ctrl.active_champion_id


def test_10_production_inference_uses_promoted_champion(temp_learning_env):
    """Requirement 10: Production inference immediately queries and uses the promoted champion."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]

    # Promote a mock candidate
    cand = LightGBMPredictor(
        model_id="lgbm_champion_promoted_v99",
        version="v99.0.0",
        n_estimators=10,
    )
    seed_X = pd.DataFrame(
        {
            "return_1": [0.005, -0.005, 0.001, -0.002],
            "rsi_14": [60, 40, 50, 45],
            "macd_hist": [0.001, -0.001, 0.0, -0.0005],
        }
    )
    cand.fit(seed_X, np.array([2, 0, 1, 0]))
    ctrl.registry.save_model(cand)
    ctrl.active_champion_id = cand.metadata.model_id
    ctrl.active_champion_model = cand

    # Verify controller serves this champion
    served_model = ctrl.get_active_champion_model()
    assert served_model is not None
    assert served_model.metadata.model_id == "lgbm_champion_promoted_v99"

    # Test that a live prediction query uses it
    test_features = pd.DataFrame([{"return_1": 0.006, "rsi_14": 65.0, "macd_hist": 0.002}])
    preds = served_model.predict(test_features)
    assert len(preds) == 1
    assert isinstance(preds[0], ModelPrediction)


def test_11_automated_production_rollback(temp_learning_env):
    """Requirement 11: Degraded production accuracy triggers automatic rollback to previous champion."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]

    # Setup champion and previous champion
    ctrl.previous_champion_id = "lgbm_champion_v1.0"
    ctrl.active_champion_id = "lgbm_challenger_degraded_v2.0"

    # Degraded production outcomes: 10 losses out of 12 predictions -> 16% win rate (< 40% threshold)
    degraded_outcomes = []
    now = datetime.now(UTC)
    for i in range(12):
        degraded_outcomes.append(
            {
                "prediction_id": f"deg_{i}",
                "model_id": ctrl.active_champion_id,
                "outcome": "LOSS" if i < 10 else "WIN",
                "brier_score": 0.45,
                "confidence": 0.70,
                "actual_return_pct": -0.8,
                "entry_price": 100.0,
                "exit_price": 99.2,
                "features_used": {"return_1": -0.008, "rsi_14": 32.0, "macd_hist": -0.001},
                "resolved_at": (now + timedelta(minutes=i)).isoformat(),
                "predicted_at": (now + timedelta(minutes=i - 15)).isoformat(),
            }
        )

    rollback_needed, reason = ctrl._check_production_rollback(degraded_outcomes)
    assert rollback_needed is True
    assert "Critical production degradation detected" in reason

    # Trigger cycle with these outcomes
    ledger = PredictionLedgerService.get_instance()
    ledger._resolved_outcomes = list(degraded_outcomes)
    res = ctrl.run_cycle_now(force=True)

    # Must execute rollback
    assert res.rollback is True
    assert ctrl.active_champion_id == ctrl.previous_champion_id


def test_12_crash_recovery_and_state_persistence(temp_learning_env):
    """Requirement 12: Application restart preserves state and recovers from interrupted cycles."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]
    state_file: Path = temp_learning_env["state_file"]
    registry: ModelRegistry = temp_learning_env["registry"]

    # Simulate an interrupted cycle during TRAINING when crash/kill occurred
    interrupted_payload = {
        "state": "TRAINING",
        "active_champion_id": "lgbm_champion_v42.0",
        "previous_champion_id": "lgbm_baseline_v41.0",
        "cycle_count": 7,
        "dataset_fingerprint": "abc123fingerprint",
        "total_eligible_outcomes": 120,
    }
    state_file.write_text(json.dumps(interrupted_payload), encoding="utf-8")

    # Restart controller instance
    new_ctrl = ContinuousLearningController(
        state_file=state_file,
        model_registry=registry,
    )

    # Clean crash recovery: state safely reset to IDLE, metrics preserved
    assert new_ctrl.state == LearningState.IDLE
    assert new_ctrl.active_champion_id == "lgbm_champion_v42.0"
    assert new_ctrl.previous_champion_id == "lgbm_baseline_v41.0"
    assert new_ctrl.cycle_count == 7
    assert new_ctrl.dataset_fingerprint == "abc123fingerprint"
    assert new_ctrl.total_eligible_outcomes == 120


def test_13_honest_statistical_evaluation_of_99_percent_target(temp_learning_env):
    """Requirement 13: Accurate statistical evaluation of 99% target with Wilson score confidence intervals."""
    ctrl: ContinuousLearningController = temp_learning_env["controller"]

    # Sub-test A: Zero resolved predictions -> NOT_YET_VERIFIABLE, no fake metrics
    rep_zero = ctrl.evaluate_99_target(outcomes=[])
    assert rep_zero.sample_size_n == 0
    assert rep_zero.directional_accuracy_pct is None
    assert rep_zero.target_99_status == Target99Status.NOT_YET_VERIFIABLE
    assert "0 resolved" in rep_zero.evaluation_summary

    # Sub-test B: Insufficient sample size (e.g. N=10 < min 30) -> NOT_YET_VERIFIABLE
    sample_10 = [
        {"prediction_id": f"p_{i}", "outcome": "WIN" if i < 7 else "LOSS", "confidence": 0.65, "actual_return_pct": 0.5, "action": "BUY"}
        for i in range(10)
    ]
    rep_small = ctrl.evaluate_99_target(outcomes=sample_10)
    assert rep_small.sample_size_n == 10
    assert rep_small.directional_accuracy_pct == 70.0
    assert rep_small.target_99_status == Target99Status.NOT_YET_VERIFIABLE
    assert rep_small.wilson_ci_lower_pct is not None
    assert rep_small.wilson_ci_upper_pct is not None
    assert rep_small.wilson_ci_lower_pct < 70.0 < rep_small.wilson_ci_upper_pct

    # Sub-test C: Statistically sufficient sample size (N=40 >= 30) but win rate < 99% -> UNMET
    sample_40 = [
        {"prediction_id": f"p_{i}", "outcome": "WIN" if i < 28 else "LOSS", "confidence": 0.68, "actual_return_pct": 0.4, "action": "BUY"}
        for i in range(40)
    ]
    rep_unmet = ctrl.evaluate_99_target(outcomes=sample_40)
    assert rep_unmet.sample_size_n == 40
    assert rep_unmet.directional_accuracy_pct == 70.0
    assert rep_unmet.target_99_status == Target99Status.UNMET
    assert rep_unmet.target_gap_pct == 29.0
    assert "UNMET" in rep_unmet.evaluation_summary

    # Sub-test D1: Small sample size (N=40, 40 wins -> 100% realized accuracy)
    # Even with 100% wins, Wilson 95% lower bound is ~91.2% (< 99.0%), so status MUST be NOT_YET_VERIFIABLE
    sample_perf_small = [
        {"prediction_id": f"p_{i}", "outcome": "WIN", "confidence": 0.90, "actual_return_pct": 1.2, "action": "BUY"}
        for i in range(40)
    ]
    rep_small_perf = ctrl.evaluate_99_target(outcomes=sample_perf_small)
    assert rep_small_perf.sample_size_n == 40
    assert rep_small_perf.directional_accuracy_pct == 100.0
    assert rep_small_perf.target_99_status == Target99Status.NOT_YET_VERIFIABLE
    assert "below the statistical requirement" in rep_small_perf.evaluation_summary
    assert rep_small_perf.wilson_ci_lower_pct is not None
    assert rep_small_perf.wilson_ci_lower_pct >= 90.0

    # Sub-test D2: Statistically defensible sample (N=400, 400 wins -> 100% accuracy)
    # With N=400, Wilson 95% CI lower bound is 99.07% >= 99.0% -> Verified MET!
    sample_perf_large = [
        {"prediction_id": f"p_large_{i}", "outcome": "WIN", "confidence": 0.92, "actual_return_pct": 1.0, "action": "BUY"}
        for i in range(400)
    ]
    rep_met = ctrl.evaluate_99_target(outcomes=sample_perf_large)
    assert rep_met.sample_size_n == 400
    assert rep_met.directional_accuracy_pct == 100.0
    assert rep_met.wilson_ci_lower_pct is not None
    assert rep_met.wilson_ci_lower_pct >= 99.0
    assert rep_met.target_99_status == Target99Status.MET
    assert "MET" in rep_met.evaluation_summary

    # Sub-test E: Abstention / NO-TRADE coverage separation
    # Model generates 10 directional BUY/SELL trades (9 wins) and 40 NO_TRADE abstentions (total N=50)
    sample_coverage = [
        {"prediction_id": f"p_dir_{i}", "outcome": "WIN" if i < 9 else "LOSS", "confidence": 0.85, "actual_return_pct": 0.8, "action": "BUY"}
        for i in range(10)
    ] + [
        {"prediction_id": f"p_neut_{i}", "outcome": "NEUTRAL", "confidence": 0.50, "actual_return_pct": 0.0, "action": "NO_TRADE"}
        for i in range(40)
    ]
    rep_cov = ctrl.evaluate_99_target(outcomes=sample_coverage)
    assert rep_cov.sample_size_n == 50
    assert rep_cov.total_samples_n == 50
    assert rep_cov.directional_samples_n == 10
    assert rep_cov.no_trade_samples_n == 40
    assert rep_cov.prediction_coverage_pct == 20.0
    assert rep_cov.abstention_rate_pct == 80.0
    assert rep_cov.directional_accuracy_pct == 90.0
    assert rep_cov.effective_accuracy_pct == 18.0  # 9 wins / 50 total = 18.0%
