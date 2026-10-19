"""Tests for Autonomous Continuous Learning Loop and Controlled Model Governance."""

import numpy as np
import pandas as pd

from learning.auditor import ModelAuditReport, ModelHealthStatus
from learning.governance import ModelGovernor
from learning.loop import ContinuousLearningCoordinator, LearningCycleResult
from models.registry import ModelRegistry


def generate_test_candles(n: int = 120) -> pd.DataFrame:
    """Generate deterministic synthetic candles for learning tests."""
    t_start = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    timestamps = [t_start + pd.Timedelta(minutes=i) for i in range(n)]

    np.random.seed(123)
    prices = 60000.0 * np.exp(np.cumsum(np.random.normal(0.0001, 0.002, n)))
    highs = prices * 1.002
    lows = prices * 0.998

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": np.full(n, 25.0),
    })


def test_retraining_trigger_healthy():
    """Verify healthy active model does not trigger retraining."""
    coordinator = ContinuousLearningCoordinator()
    healthy_audit = ModelAuditReport(
        model_id="prod_v1",
        status=ModelHealthStatus.HEALTHY,
        sample_size=30,
        win_rate=58.0,
        brier_score=0.18,
        mean_return=0.015,
        mean_error=0.08,
        buy_win_rate=60.0,
        sell_win_rate=55.0,
        retirement_recommended=False,
    )

    should_retrain, reason = coordinator.check_retraining_trigger(healthy_audit, force=False)
    assert should_retrain is False
    assert "healthy" in reason


def test_retraining_trigger_degraded():
    """Verify degraded or critical health status triggers retraining."""
    coordinator = ContinuousLearningCoordinator()
    degraded_audit = ModelAuditReport(
        model_id="prod_v1",
        status=ModelHealthStatus.CRITICAL_RETIRE,
        sample_size=30,
        win_rate=36.0,
        brier_score=0.34,
        mean_return=-0.02,
        mean_error=0.25,
        buy_win_rate=35.0,
        sell_win_rate=37.0,
        retirement_recommended=True,
    )

    should_retrain, reason = coordinator.check_retraining_trigger(degraded_audit, force=False)
    assert should_retrain is True
    assert "Critical" in reason or "degradation" in reason


def test_chronological_splits():
    """Verify chronological split guarantees zero lookahead leakage."""
    coordinator = ContinuousLearningCoordinator()
    df = generate_test_candles(100)

    train_df, oos_df = coordinator.prepare_chronological_splits(df, train_ratio=0.70)

    assert len(train_df) == 70
    assert len(oos_df) == 30
    assert train_df["timestamp"].max() < oos_df["timestamp"].min()


def test_governance_cycle_execution(tmp_path):
    """Verify complete continuous learning workflow executes under strict governance."""
    test_registry = ModelRegistry(base_dir=tmp_path / "test_models")
    test_governor = ModelGovernor(min_oos_sharpe=0.5, min_oos_win_rate=45.0)
    coordinator = ContinuousLearningCoordinator(registry=test_registry, governor=test_governor)

    df = generate_test_candles(120)

    result = coordinator.execute_governance_cycle(
        historical_candles_df=df,
        force_retrain=True,
    )

    assert isinstance(result, LearningCycleResult)
    assert result.triggered is True
    assert result.candidate_model_id is not None
    assert result.promotion_verdict is not None
    assert "status" in result.promotion_verdict
