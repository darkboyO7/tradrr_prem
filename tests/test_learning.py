"""Tests for Outcome Tracker, Model Auditor, and Model Governance Engine."""

import numpy as np
import pandas as pd
import pytest

from learning.auditor import ModelAuditor, ModelHealthStatus
from learning.governance import ModelGovernor, PromotionStatus
from learning.tracker import OutcomeTracker, ResolvedOutcome
from models.base import SignalAction
from models.statistical import StatisticalBaselineModel


def test_outcome_tracker_resolution():
    tracker = OutcomeTracker()

    # Record a prediction at $50,000 expecting Bullish outcome over 3 bars
    active = tracker.record_prediction(
        prediction_id="pred_1",
        model_id="test_model",
        symbol="BTC-USD",
        action=SignalAction.BUY,
        probability=0.75,
        expected_return=0.02,
        entry_price=50000.0,
        horizon_bars=3,
    )
    assert "pred_1" in tracker.active_predictions

    # Bar 1: Price goes to $50,500
    res1 = tracker.on_bar("BTC-USD", close=50500.0, high=50700.0, low=49900.0)
    assert len(res1) == 0  # 1 of 3 bars, not yet resolved

    # Bar 2: Price goes to $51,000
    res2 = tracker.on_bar("BTC-USD", close=51000.0, high=51200.0, low=50400.0)
    assert len(res2) == 0  # 2 of 3 bars

    # Bar 3: Price goes to $51,500 -> Horizon reached!
    res3 = tracker.on_bar("BTC-USD", close=51500.0, high=51600.0, low=50900.0)
    assert len(res3) == 1
    outcome = res3[0]

    assert outcome.prediction_id == "pred_1"
    assert outcome.outcome == "WIN"
    assert outcome.actual_return == pytest.approx(0.03, rel=1e-3)
    assert outcome.mfe > 0.03
    assert outcome.brier_score >= 0.0
    assert "pred_1" not in tracker.active_predictions


def test_model_auditor():
    auditor = ModelAuditor(min_samples=10, min_acceptable_win_rate=0.50, retire_win_rate_threshold=0.40)

    # 1. Healthy outcomes: 8 wins out of 10
    healthy_outcomes = []
    for i in range(10):
        healthy_outcomes.append(
            ResolvedOutcome(
                prediction_id=f"h_{i}",
                model_id="healthy_model",
                symbol="BTC-USD",
                action=SignalAction.BUY,
                entry_price=100.0,
                exit_price=102.0 if i < 8 else 98.0,
                actual_return=0.02 if i < 8 else -0.02,
                predicted_return=0.02,
                outcome="WIN" if i < 8 else "LOSS",
                mfe=0.025,
                mae=-0.005,
                brier_score=0.05 if i < 8 else 0.40,
                error=0.0,
                horizon_bars=5,
            )
        )

    report = auditor.audit_outcomes("healthy_model", healthy_outcomes)
    assert report.status == ModelHealthStatus.HEALTHY
    assert report.win_rate == 0.80
    assert not report.retirement_recommended

    # 2. Severely degraded outcomes: 2 wins out of 10
    degraded_outcomes = []
    for i in range(10):
        degraded_outcomes.append(
            ResolvedOutcome(
                prediction_id=f"d_{i}",
                model_id="degraded_model",
                symbol="BTC-USD",
                action=SignalAction.BUY,
                entry_price=100.0,
                exit_price=102.0 if i < 2 else 98.0,
                actual_return=0.02 if i < 2 else -0.02,
                predicted_return=0.02,
                outcome="WIN" if i < 2 else "LOSS",
                mfe=0.01,
                mae=-0.03,
                brier_score=0.45,
                error=0.04,
                horizon_bars=5,
            )
        )

    deg_report = auditor.audit_outcomes("degraded_model", degraded_outcomes)
    assert deg_report.status == ModelHealthStatus.CRITICAL_RETIRE
    assert deg_report.retirement_recommended


def test_model_governance_evaluation():
    governor = ModelGovernor(min_oos_sharpe=0.5, min_oos_win_rate=45.0, max_oos_drawdown=20.0)

    # Synthetic out of sample dataset with a bull trend
    np.random.seed(42)
    bars = 100
    closes = 50000.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.008, bars)))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC"),
        "open": closes * 0.999,
        "high": closes * 1.005,
        "low": closes * 0.995,
        "close": closes,
        "volume": [100.0] * bars,
    })

    candidate = StatisticalBaselineModel(model_id="candidate_stat")
    verdict = governor.evaluate_candidate(candidate, df)

    assert verdict.candidate_model_id == "candidate_stat"
    assert verdict.status in (PromotionStatus.PROMOTED, PromotionStatus.REJECTED)
    assert "sharpe" in verdict.candidate_metrics
    assert "gate_checks" in verdict.model_dump()
