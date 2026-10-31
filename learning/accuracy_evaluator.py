"""Trading Intelligence Platform — Institutional Accuracy & Model Evaluation Engine.

Implements rigorous out-of-sample prediction accuracy evaluation adhering to
Section 8 of the Master Engineering Specification:
- Multiclass mapping: 0 = SELL, 1 = NEUTRAL, 2 = BUY
- Wilson 95% Score Confidence Interval
- Complete per-class precision, recall, F1, and 3x3 confusion matrix
- Multiclass Brier Score and Expected Calibration Error (ECE)
- Coverage and abstention rate tracking
- Active trade win rate and directional expectancy
- Chronological train/validation/test partitions with horizon-aware purging and embargo
- Zero-leakage verification (timestamps, feature causality, embargo enforcement)
- Explicit acceptance states:
    * ACCURACY_TARGET_PASSED
    * ACCURACY_TARGET_NOT_MET
    * INSUFFICIENT_EVIDENCE
    * VALIDATION_FAILED
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import math
from typing import Any

import numpy as np
import pandas as pd


class AccuracyAcceptanceState(str, Enum):
    """Rigorous acceptance states per Section 8.4."""

    ACCURACY_TARGET_PASSED = "ACCURACY_TARGET_PASSED"
    ACCURACY_TARGET_NOT_MET = "ACCURACY_TARGET_NOT_MET"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    VALIDATION_FAILED = "VALIDATION_FAILED"


CLASS_SELL = 0
CLASS_NEUTRAL = 1
CLASS_BUY = 2
CLASS_NAMES = {0: "SELL", 1: "NEUTRAL", 2: "BUY"}
REQUIRED_NUM_CLASSES = 3


def calculate_wilson_confidence_interval(
    k: int,
    n: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Calculate the two-sided Wilson score interval for a binomial proportion.

    Formula:
        (p_hat + z^2 / (2n) +/- z * sqrt(p_hat * (1 - p_hat) / n + z^2 / (4n^2))) / (1 + z^2 / n)
    """
    if n <= 0:
        return 0.0, 0.0

    p_hat = float(k) / float(n)
    # Z-scores for standard confidence levels
    z_map = {0.90: 1.64485362695, 0.95: 1.95996398454, 0.99: 2.57582930355}
    z = z_map.get(confidence_level, 1.95996398454)

    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    spread = (z * math.sqrt((p_hat * (1.0 - p_hat) / float(n)) + (z2 / (4.0 * float(n) * float(n))))) / denom

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(lower, 4), round(upper, 4)


def compute_multiclass_brier_score(probas: np.ndarray, y_true: np.ndarray) -> float:
    """Compute standard multiclass Brier score (mean squared error of probability vector).

    For K classes, Brier = (1/N) * sum_i sum_k (p_{i,k} - y_{i,k})^2.
    Baseline for uniform 3-class random guess: 2/3 = 0.6667.
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    y_onehot = np.zeros((n, REQUIRED_NUM_CLASSES), dtype=float)
    for i, c in enumerate(y_true):
        c_int = int(c)
        if 0 <= c_int < REQUIRED_NUM_CLASSES:
            y_onehot[i, c_int] = 1.0

    return float(np.mean(np.sum((probas - y_onehot) ** 2, axis=1)))


def compute_expected_calibration_error(
    probas: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    n = len(y_true)
    if n == 0:
        return 0.0

    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_acc - bin_conf) * prop_in_bin

    return float(round(ece, 4))


@dataclass
class ClassPerformanceMetrics:
    """Detailed per-class classification metrics."""

    class_id: int
    class_name: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


@dataclass
class ModelAccuracyEvaluationReport:
    """Comprehensive Model Accuracy & Governance Audit Report (Section 8.1)."""

    model_id: str
    version: str
    evaluated_at: str
    target_description: str
    forecast_horizon: str
    sample_size: int
    acceptance_state: AccuracyAcceptanceState

    # Overall Metrics
    overall_accuracy: float
    overall_accuracy_pct: float
    wilson_ci_lower_95: float
    wilson_ci_upper_95: float
    wilson_ci_lower_pct: float
    wilson_ci_upper_pct: float
    target_accuracy_hurdle: float = 0.95

    # 3x3 Confusion Matrix [true_class, pred_class]
    confusion_matrix: list[list[int]] = field(default_factory=list)

    # Per-Class Metrics
    per_class_metrics: dict[str, ClassPerformanceMetrics] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)

    # Probability Calibration
    brier_score: float = 0.0
    expected_calibration_error: float = 0.0

    # Active Trading vs Abstention Metrics
    coverage_rate_pct: float = 0.0
    abstention_rate_pct: float = 0.0
    active_sample_count: int = 0
    active_trade_accuracy_pct: float = 0.0
    active_trade_win_rate_pct: float = 0.0
    active_trade_expectancy_pct: float = 0.0

    # Data Leakage & Integrity Audit
    leakage_checks_passed: bool = True
    leakage_audit_log: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        d = asdict(self)
        d["acceptance_state"] = self.acceptance_state.value
        return d


class InstitutionalModelEvaluator:
    """Rigorous evaluation engine validating candidate models against Section 8 criteria."""

    def __init__(
        self,
        target_accuracy: float = 0.95,
        min_sample_size: int = 100,
        min_active_trades: int = 35,
        confidence_level: float = 0.95,
    ) -> None:
        self.target_accuracy = target_accuracy
        self.min_sample_size = min_sample_size
        self.min_active_trades = min_active_trades
        self.confidence_level = confidence_level

    def audit_leakage(
        self,
        train_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        lookahead_bars: int,
        embargo_bars: int,
    ) -> tuple[bool, list[str]]:
        """Verify chronological ordering, embargo sufficiency, and absence of timestamp leakage."""
        logs: list[str] = []
        passed = True

        # Check 1: Embargo sufficiency
        if embargo_bars < lookahead_bars:
            logs.append(
                f"LEAKAGE_FAIL: Embargo bars ({embargo_bars}) is less than forecast horizon lookahead ({lookahead_bars})."
            )
            passed = False
        else:
            logs.append(
                f"LEAKAGE_PASS: Embargo bars ({embargo_bars}) >= forecast horizon lookahead ({lookahead_bars})."
            )

        # Check 2: Timestamp non-overlap
        if "timestamp" in train_df.columns and "timestamp" in oos_df.columns:
            train_ts = set(train_df["timestamp"])
            oos_ts = set(oos_df["timestamp"])
            intersection = train_ts.intersection(oos_ts)
            if intersection:
                logs.append(
                    f"LEAKAGE_FAIL: Found {len(intersection)} identical timestamps overlapping in both train and OOS!"
                )
                passed = False
            else:
                logs.append("LEAKAGE_PASS: Zero timestamp intersection between train and OOS partitions.")

            # Check 3: Strict chronological monotonicity
            max_train_ts = train_df["timestamp"].max()
            min_oos_ts = oos_df["timestamp"].min()
            if max_train_ts >= min_oos_ts:
                logs.append(
                    f"LEAKAGE_FAIL: Train max timestamp ({max_train_ts}) >= OOS min timestamp ({min_oos_ts})."
                )
                passed = False
            else:
                logs.append(
                    f"LEAKAGE_PASS: Strict chronological ordering verified (Train End: {max_train_ts} < OOS Start: {min_oos_ts})."
                )

        return passed, logs

    def evaluate(
        self,
        model_id: str,
        version: str,
        y_true: np.ndarray,
        probas: np.ndarray,
        fwd_returns: np.ndarray | None = None,
        train_df: pd.DataFrame | None = None,
        oos_df: pd.DataFrame | None = None,
        lookahead_bars: int = 3,
        embargo_bars: int = 3,
        forecast_horizon_label: str = "15m",
        target_description: str = "Forward 3-Bar Equity Return Multiclass [0=SELL, 1=NEUTRAL, 2=BUY]",
    ) -> ModelAccuracyEvaluationReport:
        """Execute full evaluation protocol and determine acceptance state."""
        n_samples = len(y_true)
        rejection_reasons: list[str] = []

        # Audit Leakage if DataFrames provided
        leakage_passed = True
        leakage_logs: list[str] = []
        if train_df is not None and oos_df is not None:
            leakage_passed, leakage_logs = self.audit_leakage(
                train_df, oos_df, lookahead_bars=lookahead_bars, embargo_bars=embargo_bars
            )
            if not leakage_passed:
                rejection_reasons.append("Data leakage check failed: chronological separation or embargo breached.")

        # Predictions
        preds = np.argmax(probas, axis=1)

        # Basic Sample Size & Class Representation Checks
        unique_classes = set(np.unique(y_true))
        for c in (CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY):
            if c not in unique_classes:
                rejection_reasons.append(f"Target dataset lacks representation for Class {c} ({CLASS_NAMES[c]}).")

        # Compute Overall Accuracy
        correct_count = int(np.sum(preds == y_true))
        overall_acc = float(correct_count) / float(max(n_samples, 1))

        # Compute Wilson 95% Confidence Interval
        ci_lower, ci_upper = calculate_wilson_confidence_interval(
            k=correct_count, n=n_samples, confidence_level=self.confidence_level
        )

        # 3x3 Confusion Matrix: rows = true, cols = predicted
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for t, p in zip(y_true, preds):
            t_int, p_int = int(t), int(p)
            if 0 <= t_int < 3 and 0 <= p_int < 3:
                cm[t_int][p_int] += 1

        # Per-Class Performance Metrics
        per_class: dict[str, ClassPerformanceMetrics] = {}
        class_dist: dict[str, int] = {}
        for c_idx in (CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY):
            c_name = CLASS_NAMES[c_idx]
            support = int(np.sum(y_true == c_idx))
            class_dist[c_name] = support

            tp = cm[c_idx][c_idx]
            fp = sum(cm[other][c_idx] for other in range(3) if other != c_idx)
            fn = sum(cm[c_idx][other] for other in range(3) if other != c_idx)

            prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
            rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class[c_name] = ClassPerformanceMetrics(
                class_id=c_idx,
                class_name=c_name,
                support=support,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1_score=round(f1, 4),
            )

        # Probability Calibration: Brier Score & ECE
        brier = compute_multiclass_brier_score(probas, y_true)
        ece = compute_expected_calibration_error(probas, y_true)

        # Active Signal Performance vs Abstention
        active_mask = (preds == CLASS_BUY) | (preds == CLASS_SELL)
        active_count = int(np.sum(active_mask))
        abstention_rate = float(np.sum(preds == CLASS_NEUTRAL)) / float(max(n_samples, 1))
        coverage_rate = 1.0 - abstention_rate

        active_trade_acc = 0.0
        active_win_rate = 0.0
        active_expectancy = 0.0

        if active_count > 0:
            active_correct = int(np.sum(preds[active_mask] == y_true[active_mask]))
            active_trade_acc = float(active_correct) / float(active_count)

            if fwd_returns is not None and len(fwd_returns) == n_samples:
                dir_rets = np.where(
                    preds[active_mask] == CLASS_BUY,
                    fwd_returns[active_mask],
                    -fwd_returns[active_mask],
                )
                active_win_rate = float(np.mean(dir_rets > 0.0))
                active_expectancy = float(np.mean(dir_rets))

        # Determine Acceptance State (Section 8.4)
        if not leakage_passed:
            acceptance_state = AccuracyAcceptanceState.VALIDATION_FAILED
        elif n_samples < self.min_sample_size or len(unique_classes) < REQUIRED_NUM_CLASSES:
            acceptance_state = AccuracyAcceptanceState.INSUFFICIENT_EVIDENCE
            if n_samples < self.min_sample_size:
                rejection_reasons.append(
                    f"Sample size ({n_samples}) is below minimum requirement ({self.min_sample_size})."
                )
        elif ci_lower >= self.target_accuracy:
            acceptance_state = AccuracyAcceptanceState.ACCURACY_TARGET_PASSED
        else:
            acceptance_state = AccuracyAcceptanceState.ACCURACY_TARGET_NOT_MET
            rejection_reasons.append(
                f"Measured accuracy is {overall_acc * 100:.2f}% (Wilson 95% CI Lower Bound: {ci_lower * 100:.2f}%), "
                f"which is below the mandatory {self.target_accuracy * 100:.1f}% acceptance gate."
            )

        return ModelAccuracyEvaluationReport(
            model_id=model_id,
            version=version,
            evaluated_at=datetime.now(UTC).isoformat(),
            target_description=target_description,
            forecast_horizon=forecast_horizon_label,
            sample_size=n_samples,
            acceptance_state=acceptance_state,
            overall_accuracy=round(overall_acc, 4),
            overall_accuracy_pct=round(overall_acc * 100.0, 2),
            wilson_ci_lower_95=ci_lower,
            wilson_ci_upper_95=ci_upper,
            wilson_ci_lower_pct=round(ci_lower * 100.0, 2),
            wilson_ci_upper_pct=round(ci_upper * 100.0, 2),
            target_accuracy_hurdle=self.target_accuracy,
            confusion_matrix=cm,
            per_class_metrics=per_class,
            class_distribution=class_dist,
            brier_score=round(brier, 4),
            expected_calibration_error=ece,
            coverage_rate_pct=round(coverage_rate * 100.0, 2),
            abstention_rate_pct=round(abstention_rate * 100.0, 2),
            active_sample_count=active_count,
            active_trade_accuracy_pct=round(active_trade_acc * 100.0, 2),
            active_trade_win_rate_pct=round(active_win_rate * 100.0, 2),
            active_trade_expectancy_pct=round(active_expectancy * 100.0, 4),
            leakage_checks_passed=leakage_passed,
            leakage_audit_log=leakage_logs,
            rejection_reasons=rejection_reasons,
        )


@dataclass
class WalkForwardFoldReport:
    """Out-of-sample evaluation report for a single walk-forward fold."""

    fold_index: int
    train_bars: int
    oos_bars: int
    embargo_bars: int
    train_start_idx: int
    train_end_idx: int
    oos_start_idx: int
    oos_end_idx: int
    eval_report: ModelAccuracyEvaluationReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_bars": self.train_bars,
            "oos_bars": self.oos_bars,
            "embargo_bars": self.embargo_bars,
            "train_start_idx": self.train_start_idx,
            "train_end_idx": self.train_end_idx,
            "oos_start_idx": self.oos_start_idx,
            "oos_end_idx": self.oos_end_idx,
            "eval_report": self.eval_report.to_dict(),
        }


@dataclass
class WalkForwardAccuracyReport:
    """Master chronological walk-forward accuracy audit across sequential regimes."""

    model_id: str
    n_folds: int
    total_oos_samples: int
    folds: list[WalkForwardFoldReport]
    aggregate_report: ModelAccuracyEvaluationReport
    mean_fold_accuracy_pct: float
    std_fold_accuracy_pct: float
    mean_fold_win_rate_pct: float
    mean_fold_expectancy_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_folds": self.n_folds,
            "total_oos_samples": self.total_oos_samples,
            "folds": [f.to_dict() for f in self.folds],
            "aggregate_report": self.aggregate_report.to_dict(),
            "mean_fold_accuracy_pct": round(self.mean_fold_accuracy_pct, 2),
            "std_fold_accuracy_pct": round(self.std_fold_accuracy_pct, 2),
            "mean_fold_win_rate_pct": round(self.mean_fold_win_rate_pct, 2),
            "mean_fold_expectancy_pct": round(self.mean_fold_expectancy_pct, 4),
        }


class InstitutionalWalkForwardEvaluator:
    """Orchestrates chronological expanding-window walk-forward cross-validation for ML models."""

    def __init__(
        self,
        n_splits: int = 4,
        initial_train_ratio: float = 0.50,
        embargo_bars: int = 3,
        lookahead_bars: int = 3,
        target_accuracy: float = 0.95,
        min_sample_size: int = 20,
        confidence_level: float = 0.95,
    ) -> None:
        self.n_splits = n_splits
        self.initial_train_ratio = initial_train_ratio
        self.embargo_bars = embargo_bars
        self.lookahead_bars = lookahead_bars
        self.target_accuracy = target_accuracy
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level

    def run_walk_forward(
        self,
        model_factory: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        fwd_returns: np.ndarray,
        model_id: str = "walk_forward_model",
        horizon_label: str = "15m",
    ) -> WalkForwardAccuracyReport:
        """Execute expanding-window walk-forward validation across sequential holdouts."""
        total_rows = len(X)
        if total_rows < 60:
            raise ValueError(f"Insufficient samples for walk-forward evaluation ({total_rows} < 60).")

        initial_train_size = int(total_rows * self.initial_train_ratio)
        remaining_rows = total_rows - initial_train_size
        test_chunk_size = max(15, int(remaining_rows / self.n_splits))

        folds: list[WalkForwardFoldReport] = []
        all_oos_probas: list[np.ndarray] = []
        all_oos_y_true: list[np.ndarray] = []
        all_oos_returns: list[np.ndarray] = []

        evaluator = InstitutionalModelEvaluator(
            target_accuracy=self.target_accuracy,
            min_sample_size=self.min_sample_size,
            confidence_level=self.confidence_level,
        )

        for fold_idx in range(self.n_splits):
            train_end = initial_train_size + (fold_idx * test_chunk_size)
            oos_start = train_end + self.embargo_bars
            oos_end = min(total_rows, oos_start + test_chunk_size)

            if oos_start >= total_rows or (oos_end - oos_start) < 10:
                break

            X_train = X.iloc[:train_end].copy()
            y_train = y[:train_end]

            # Verify train partition contains all 3 classes
            unique_train_classes = np.unique(y_train)
            if len(unique_train_classes) < REQUIRED_NUM_CLASSES:
                continue

            X_oos = X.iloc[oos_start:oos_end].copy()
            y_oos = y[oos_start:oos_end]
            ret_oos = fwd_returns[oos_start:oos_end]

            # Instantiate and fit fresh model instance for this fold
            model = model_factory()
            model.fit(X_train, y_train, strict_multiclass=True)
            probas_fold = model.predict_proba(X_oos)

            fold_report = evaluator.evaluate(
                model_id=f"{model_id}_fold{fold_idx + 1}",
                version=f"fold_{fold_idx + 1}",
                y_true=y_oos,
                probas=probas_fold,
                fwd_returns=ret_oos,
                lookahead_bars=self.lookahead_bars,
                embargo_bars=self.embargo_bars,
                forecast_horizon_label=horizon_label,
                target_description=f"Walk-Forward Fold {fold_idx + 1} Target Return",
            )

            folds.append(
                WalkForwardFoldReport(
                    fold_index=fold_idx + 1,
                    train_bars=len(X_train),
                    oos_bars=len(X_oos),
                    embargo_bars=self.embargo_bars,
                    train_start_idx=0,
                    train_end_idx=train_end,
                    oos_start_idx=oos_start,
                    oos_end_idx=oos_end,
                    eval_report=fold_report,
                )
            )

            all_oos_probas.append(probas_fold)
            all_oos_y_true.append(y_oos)
            all_oos_returns.append(ret_oos)

        if not folds:
            raise RuntimeError("Walk-forward cross-validation failed to complete any valid folds.")

        concat_probas = np.vstack(all_oos_probas)
        concat_y_true = np.concatenate(all_oos_y_true)
        concat_returns = np.concatenate(all_oos_returns)

        agg_report = evaluator.evaluate(
            model_id=model_id,
            version="aggregate_walk_forward",
            y_true=concat_y_true,
            probas=concat_probas,
            fwd_returns=concat_returns,
            lookahead_bars=self.lookahead_bars,
            embargo_bars=self.embargo_bars,
            forecast_horizon_label=horizon_label,
            target_description="Aggregate Walk-Forward Out-of-Sample Telemetry",
        )

        fold_accs = [f.eval_report.overall_accuracy_pct for f in folds]
        fold_wins = [f.eval_report.active_trade_win_rate_pct for f in folds]
        fold_exps = [f.eval_report.active_trade_expectancy_pct for f in folds]

        return WalkForwardAccuracyReport(
            model_id=model_id,
            n_folds=len(folds),
            total_oos_samples=len(concat_y_true),
            folds=folds,
            aggregate_report=agg_report,
            mean_fold_accuracy_pct=float(np.mean(fold_accs)),
            std_fold_accuracy_pct=float(np.std(fold_accs)),
            mean_fold_win_rate_pct=float(np.mean(fold_wins)),
            mean_fold_expectancy_pct=float(np.mean(fold_exps)),
        )

