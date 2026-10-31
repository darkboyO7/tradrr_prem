"""Unit tests for the Institutional Accuracy & Model Evaluation Engine (Section 8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from learning.accuracy_evaluator import (
    CLASS_BUY,
    CLASS_NAMES,
    CLASS_NEUTRAL,
    CLASS_SELL,
    AccuracyAcceptanceState,
    InstitutionalModelEvaluator,
    InstitutionalWalkForwardEvaluator,
    WalkForwardAccuracyReport,
    WalkForwardFoldReport,
    calculate_wilson_confidence_interval,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)


def test_wilson_confidence_interval_bounds() -> None:
    """Verify Wilson score confidence interval calculation against statistical properties."""
    # Zero sample
    assert calculate_wilson_confidence_interval(0, 0) == (0.0, 0.0)

    # 50 out of 100 -> center should be 0.50, lower ~0.40, upper ~0.60
    lower, upper = calculate_wilson_confidence_interval(50, 100)
    assert 0.39 <= lower <= 0.41
    assert 0.59 <= upper <= 0.61
    assert lower < 0.50 < upper

    # 95 out of 100 -> lower ~0.887, upper ~0.983
    lower_95, upper_95 = calculate_wilson_confidence_interval(95, 100)
    assert 0.88 <= lower_95 <= 0.90
    assert 0.97 <= upper_95 <= 0.99
    # Crucially: Even with 95% observed accuracy on n=100, the lower bound is ~89% (< 95%)!
    assert lower_95 < 0.95

    # 990 out of 1000 -> lower ~0.981, upper ~0.995 (passes 95% gate)
    lower_large, upper_large = calculate_wilson_confidence_interval(990, 1000)
    assert lower_large >= 0.97
    assert upper_large <= 1.0


def test_multiclass_brier_score() -> None:
    """Verify Brier score on perfect predictions vs random guessing."""
    # Perfect predictions: probability 1.0 on true class
    probas_perfect = np.array([
        [1.0, 0.0, 0.0],  # true 0
        [0.0, 1.0, 0.0],  # true 1
        [0.0, 0.0, 1.0],  # true 2
    ])
    y_true = np.array([0, 1, 2])
    brier_perfect = compute_multiclass_brier_score(probas_perfect, y_true)
    assert brier_perfect == 0.0

    # Uniform random guess (1/3 for each class): Brier = (2/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 6/9 = 0.6667
    probas_uniform = np.array([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    brier_uniform = compute_multiclass_brier_score(probas_uniform, y_true)
    assert pytest.approx(brier_uniform, abs=1e-4) == 0.6667


def test_expected_calibration_error() -> None:
    """Verify Expected Calibration Error (ECE) metric."""
    probas = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.9, 0.05, 0.05],
    ])
    y_true = np.array([0, 1, 2, 0])
    ece = compute_expected_calibration_error(probas, y_true)
    assert 0.0 <= ece <= 1.0


def test_evaluator_leakage_audit() -> None:
    """Verify data leakage auditor detects embargo breaches and timestamp overlap."""
    evaluator = InstitutionalModelEvaluator()

    # Case 1: Valid chronological ordering with sufficient embargo
    train_df = pd.DataFrame({"timestamp": pd.date_range("2026-10-01", periods=100, freq="5min")})
    oos_df = pd.DataFrame({"timestamp": pd.date_range("2026-10-02", periods=50, freq="5min")})

    passed, logs = evaluator.audit_leakage(train_df, oos_df, lookahead_bars=3, embargo_bars=3)
    assert passed is True
    assert any("LEAKAGE_PASS" in log for log in logs)

    # Case 2: Insufficient embargo (embargo_bars < lookahead_bars)
    passed_emb, logs_emb = evaluator.audit_leakage(train_df, oos_df, lookahead_bars=5, embargo_bars=2)
    assert passed_emb is False
    assert any("Embargo bars" in log for log in logs_emb)

    # Case 3: Overlapping timestamps
    train_leak = pd.DataFrame({"timestamp": [1, 2, 3, 4, 5]})
    oos_leak = pd.DataFrame({"timestamp": [5, 6, 7, 8, 9]})
    passed_ts, logs_ts = evaluator.audit_leakage(train_leak, oos_leak, lookahead_bars=1, embargo_bars=1)
    assert passed_ts is False
    assert any("overlapping" in log for log in logs_ts)


def test_evaluator_insufficient_evidence_state() -> None:
    """Verify INSUFFICIENT_EVIDENCE state is triggered on small sample size."""
    evaluator = InstitutionalModelEvaluator(min_sample_size=100)

    # n = 40 (< 100)
    y_true = np.array([0] * 15 + [1] * 15 + [2] * 10)
    probas = np.zeros((40, 3))
    for i, c in enumerate(y_true):
        probas[i, c] = 0.9
        probas[i, (c + 1) % 3] = 0.05
        probas[i, (c + 2) % 3] = 0.05

    report = evaluator.evaluate(
        model_id="candidate_small",
        version="v1",
        y_true=y_true,
        probas=probas,
    )
    assert report.acceptance_state == AccuracyAcceptanceState.INSUFFICIENT_EVIDENCE
    assert any("below minimum requirement" in r for r in report.rejection_reasons)


def test_evaluator_accuracy_target_not_met_state() -> None:
    """Verify ACCURACY_TARGET_NOT_MET state when accuracy is below 95%."""
    evaluator = InstitutionalModelEvaluator(target_accuracy=0.95, min_sample_size=100)

    # n = 150 with ~60% accuracy
    np.random.seed(42)
    y_true = np.random.choice([0, 1, 2], size=150)
    preds = y_true.copy()
    # Flip 40% to wrong
    flip_indices = np.random.choice(150, size=60, replace=False)
    for idx in flip_indices:
        preds[idx] = (preds[idx] + 1) % 3

    probas = np.zeros((150, 3))
    for i, p in enumerate(preds):
        probas[i, p] = 0.70
        probas[i, (p + 1) % 3] = 0.15
        probas[i, (p + 2) % 3] = 0.15

    report = evaluator.evaluate(
        model_id="candidate_moderate",
        version="v1",
        y_true=y_true,
        probas=probas,
        lookahead_bars=3,
        embargo_bars=3,
    )
    assert report.acceptance_state == AccuracyAcceptanceState.ACCURACY_TARGET_NOT_MET
    assert report.overall_accuracy < 0.95
    assert report.wilson_ci_lower_95 < 0.95
    assert any("below the mandatory 95.0%" in r for r in report.rejection_reasons)


def test_evaluator_accuracy_target_passed_state() -> None:
    """Verify ACCURACY_TARGET_PASSED state when Wilson CI lower bound exceeds 95% on large sample."""
    evaluator = InstitutionalModelEvaluator(target_accuracy=0.95, min_sample_size=100)

    # n = 500 with 98.4% accuracy (492/500 correct) -> Wilson CI lower bound is ~0.968 >= 0.95
    n = 500
    y_true = np.array([0] * 160 + [1] * 170 + [2] * 170)
    preds = y_true.copy()
    # Only 8 errors
    for idx in range(8):
        preds[idx] = (preds[idx] + 1) % 3

    probas = np.zeros((n, 3))
    for i, p in enumerate(preds):
        probas[i, p] = 0.92
        probas[i, (p + 1) % 3] = 0.04
        probas[i, (p + 2) % 3] = 0.04

    report = evaluator.evaluate(
        model_id="candidate_exceptional",
        version="v1",
        y_true=y_true,
        probas=probas,
        lookahead_bars=3,
        embargo_bars=3,
    )
    assert report.acceptance_state == AccuracyAcceptanceState.ACCURACY_TARGET_PASSED
    assert report.overall_accuracy >= 0.98
    assert report.wilson_ci_lower_95 >= 0.95
    assert len(report.rejection_reasons) == 0


def test_per_class_metrics_and_confusion_matrix() -> None:
    """Verify confusion matrix and per-class precision, recall, and F1 calculations."""
    evaluator = InstitutionalModelEvaluator()

    # Hand-crafted small example:
    # True: [0, 0, 1, 1, 2, 2]
    # Pred: [0, 1, 1, 2, 2, 2]
    y_true = np.array([CLASS_SELL, CLASS_SELL, CLASS_NEUTRAL, CLASS_NEUTRAL, CLASS_BUY, CLASS_BUY])
    probas = np.array([
        [0.8, 0.1, 0.1],  # pred 0 (true 0) -> TP for 0
        [0.2, 0.7, 0.1],  # pred 1 (true 0) -> FN for 0, FP for 1
        [0.1, 0.8, 0.1],  # pred 1 (true 1) -> TP for 1
        [0.1, 0.1, 0.8],  # pred 2 (true 1) -> FN for 1, FP for 2
        [0.1, 0.1, 0.8],  # pred 2 (true 2) -> TP for 2
        [0.1, 0.1, 0.8],  # pred 2 (true 2) -> TP for 2
    ])

    report = evaluator.evaluate(
        model_id="toy_model",
        version="v0",
        y_true=y_true,
        probas=probas,
    )

    # CM: rows = true, cols = pred
    # True 0: pred 0=1, pred 1=1, pred 2=0
    # True 1: pred 0=0, pred 1=1, pred 2=1
    # True 2: pred 0=0, pred 1=0, pred 2=2
    assert report.confusion_matrix == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 2],
    ]

    # Check BUY class metrics (class 2)
    # TP = 2, FP = 1 (from true 1), FN = 0
    # Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0
    buy_m = report.per_class_metrics["BUY"]
    assert buy_m.true_positives == 2
    assert buy_m.false_positives == 1
    assert buy_m.false_negatives == 0
    assert pytest.approx(buy_m.precision, abs=1e-4) == 0.6667
    assert buy_m.recall == 1.0


def test_institutional_walk_forward_evaluator() -> None:
    """Verify chronological walk-forward cross-validation across sequential holdout folds."""
    np.random.seed(42)
    n = 150
    # Generate 150 samples with 3 features
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    # Target 3 classes (0, 1, 2)
    y = np.random.choice([CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY], size=n)
    fwd_ret = np.random.normal(0.0005, 0.005, n)

    class DummyModel:
        def fit(self, X_t: pd.DataFrame, y_t: np.ndarray, strict_multiclass: bool = True) -> None:
            self.classes_ = np.array([0, 1, 2])

        def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
            k = len(X_test)
            # Return valid probability simplex
            p = np.full((k, 3), 1.0 / 3.0)
            return p

    wf = InstitutionalWalkForwardEvaluator(
        n_splits=3,
        initial_train_ratio=0.50,
        embargo_bars=2,
        lookahead_bars=2,
        min_sample_size=15,
    )
    report = wf.run_walk_forward(
        model_factory=lambda: DummyModel(),
        X=X,
        y=y,
        fwd_returns=fwd_ret,
        model_id="test_dummy_model",
    )

    assert isinstance(report, WalkForwardAccuracyReport)
    assert report.n_folds == 3
    assert len(report.folds) == 3
    assert report.total_oos_samples > 0
    assert report.aggregate_report.sample_size == report.total_oos_samples

    # Check serialized dictionary structure
    d = report.to_dict()
    assert d["model_id"] == "test_dummy_model"
    assert d["n_folds"] == 3
    assert len(d["folds"]) == 3
    assert "aggregate_report" in d
    assert "mean_fold_accuracy_pct" in d

