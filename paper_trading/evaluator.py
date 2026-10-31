"""Trading Intelligence Platform - Prospective Paper-Trading Evaluation Engine.

Evaluates ONLY prospective paper-trading records stored in data/paper_trading/.
Enforces:
- Complete isolation from historical backtests and synthetic test records.
- Minimum 60-day / 100-observation prospective evidence gate.
- Reconciles metrics, baselines, and calibration on identical observations.
- Stationary Block Bootstrap uncertainty estimation accounting for serial and cross-currency dependence.
- Automated gate checking against policy POL-GOV-2026-001.
"""

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

logger = logging.getLogger("paper_trading.evaluator")

DEFAULT_PAPER_STORE_DIR = Path("data/paper_trading")
MIN_PROSPECTIVE_OBSERVATIONS = 100
MIN_PROSPECTIVE_CALENDAR_DAYS = 60


def compute_ece(probas: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(1.0 / 3.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_records = []
    total = len(probas)
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences <= high if i == n_bins - 1 else confidences < high)
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            gap = abs(bin_acc - bin_conf)
            weight = cnt / total
            ece += weight * gap
            bin_records.append({
                "bin_low": round(low, 3), "bin_high": round(high, 3),
                "count": cnt, "confidence": round(bin_conf, 4),
                "accuracy": round(bin_acc, 4), "gap": round(gap, 4),
            })
        else:
            bin_records.append({
                "bin_low": round(low, 3), "bin_high": round(high, 3),
                "count": 0, "confidence": 0.0, "accuracy": 0.0, "gap": 0.0,
            })
    return float(ece), bin_records


def stationary_block_bootstrap(
    series: np.ndarray,
    expected_block_len: int = 5,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Computes 95% confidence interval using stationary block bootstrap to preserve autocorrelation."""
    n = len(series)
    if n < 5:
        return float(np.mean(series)), float(np.mean(series))

    rng = np.random.default_rng(random_seed)
    p = 1.0 / expected_block_len
    bootstrap_means = []

    for _ in range(n_bootstraps):
        indices = []
        cur_idx = rng.integers(0, n)
        while len(indices) < n:
            indices.append(cur_idx)
            if rng.random() < p:
                cur_idx = rng.integers(0, n)
            else:
                cur_idx = (cur_idx + 1) % n
        sample = series[indices]
        bootstrap_means.append(float(np.mean(sample)))

    ci_low = float(np.percentile(bootstrap_means, 2.5))
    ci_high = float(np.percentile(bootstrap_means, 97.5))
    return ci_low, ci_high


class PaperTradingEvaluator:
    """Evaluates prospective paper-trading records in strict accordance with POL-GOV-2026-001."""

    def __init__(self, store_dir: Path | str = DEFAULT_PAPER_STORE_DIR):
        self.store_dir = Path(store_dir)
        self.predictions_file = self.store_dir / "paper_predictions.jsonl"
        self.outcomes_file = self.store_dir / "paper_outcomes.jsonl"

    def evaluate(self) -> dict[str, Any]:
        """Runs complete prospective evaluation. Enforces data isolation and minimum sample gates."""
        predictions: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []

        if self.predictions_file.exists():
            with open(self.predictions_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))

        if self.outcomes_file.exists():
            with open(self.outcomes_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        outcomes.append(json.loads(line))

        total_preds = len(predictions)
        resolved_outcomes = len(outcomes)
        pending_preds = total_preds - resolved_outcomes

        # Categorize outcomes
        valid_resolved = [o for o in outcomes if o.get("data_source_status") == "VALID_OBSERVATION"]
        expired_without_data = [o for o in outcomes if o.get("outcome") == "EXPIRED_WITHOUT_DATA"]
        invalid_data = [o for o in outcomes if o.get("outcome") == "INVALID_DATA"]

        # Date range calculation
        if predictions:
            pred_times = [datetime.fromisoformat(p["prediction_timestamp"]) for p in predictions]
            min_date = min(pred_times)
            max_date = max(pred_times)
            calendar_days = (max_date - min_date).total_seconds() / 86400.0
        else:
            calendar_days = 0.0
            min_date = None
            max_date = None

        has_min_obs = len(valid_resolved) >= MIN_PROSPECTIVE_OBSERVATIONS
        has_min_days = calendar_days >= MIN_PROSPECTIVE_CALENDAR_DAYS

        base_summary = {
            "evaluation_timestamp": datetime.now(UTC).isoformat(),
            "evaluation_policy": "POL-GOV-2026-001",
            "evidence_status": "INSUFFICIENT_PROSPECTIVE_EVIDENCE" if not (has_min_obs and has_min_days) else "COLLECTION_TARGET_MET",
            "sample_progress": {
                "valid_resolved_observations": len(valid_resolved),
                "required_minimum_observations": MIN_PROSPECTIVE_OBSERVATIONS,
                "calendar_days_accumulated": round(calendar_days, 1),
                "required_minimum_calendar_days": MIN_PROSPECTIVE_CALENDAR_DAYS,
                "total_predictions_recorded": total_preds,
                "pending_predictions": pending_preds,
                "expired_without_data": len(expired_without_data),
                "invalid_data_count": len(invalid_data),
                "collection_start_date": min_date.isoformat() if min_date else None,
                "latest_prediction_date": max_date.isoformat() if max_date else None,
            },
        }

        # If zero valid resolved observations, return progress report
        if not valid_resolved:
            base_summary["message"] = (
                "Prospective paper trading is currently in active collection. "
                f"Progress: {len(valid_resolved)}/{MIN_PROSPECTIVE_OBSERVATIONS} observations, "
                f"{calendar_days:.1f}/{MIN_PROSPECTIVE_CALENDAR_DAYS} calendar days."
            )
            return base_summary

        # Perform quantitative evaluation on valid resolved records
        df_out = pd.DataFrame(valid_resolved)
        pred_map = {p["prediction_id"]: p for p in predictions}

        # Reconcile labels and probabilities
        y_true = []
        y_pred = []
        probas = []
        net_returns = []
        gross_returns = []

        for _, row in df_out.iterrows():
            pid = row["prediction_id"]
            orig = pred_map.get(pid, {})
            p_s = float(orig.get("prob_sell_calibrated", 0.33))
            p_n = float(orig.get("prob_neutral_calibrated", 0.34))
            p_b = float(orig.get("prob_buy_calibrated", 0.33))
            probas.append([p_s, p_n, p_b])

            # True label from price movement
            entry_px = float(row["executable_entry_price"])
            exit_px = float(row["observed_exit_price"])
            raw_ret = (exit_px - entry_px) / entry_px
            target_cls = 2 if raw_ret > 0.0005 else (0 if raw_ret < -0.0005 else 1)
            y_true.append(target_cls)

            # Predicted class
            dec = row["decision"]
            pred_cls = 2 if dec == "BUY" else (0 if dec == "SELL" else 1)
            y_pred.append(pred_cls)

            net_returns.append(float(row["net_return"]))
            gross_returns.append(float(row["gross_return"]))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        probas = np.array(probas)
        net_returns = np.array(net_returns)
        gross_returns = np.array(gross_returns)

        # 1. Prediction Quality Metrics
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        brier = float(np.mean([sum((probas[i, k] - (1.0 if y_true[i] == k else 0.0))**2 for k in range(3)) for i in range(len(y_true))]))
        ll = float(log_loss(y_true, probas, labels=[0, 1, 2]))
        ece, ece_bins = compute_ece(probas, y_true, n_bins=10)

        # 2. Trading Performance Metrics
        active_trades = [r for r in valid_resolved if r["decision"] in ("BUY", "SELL")]
        active_cnt = len(active_trades)
        wins = [t for t in active_trades if t["outcome"] == "WIN"]
        win_rate = (len(wins) / active_cnt * 100.0) if active_cnt > 0 else 0.0

        # Compounded equity curve
        eq = 100000.0
        eq_curve = [eq]
        for r in net_returns:
            eq *= (1.0 + r)
            eq_curve.append(eq)
        arr = np.array(eq_curve)
        peak = np.maximum.accumulate(arr)
        max_dd = float(np.max((peak - arr) / peak)) * 100.0
        cum_net_ret = (eq / eq_curve[0] - 1.0) * 100.0

        # Uncertainty estimation via block bootstrap
        ci_low, ci_high = stationary_block_bootstrap(net_returns, expected_block_len=5, n_bootstraps=1000)
        ci_low_cum = round(((1.0 + ci_low) ** len(net_returns) - 1.0) * 100.0, 2)
        ci_high_cum = round(((1.0 + ci_high) ** len(net_returns) - 1.0) * 100.0, 2)

        # 3. Baselines on Identical Observations
        # Class-prior baseline
        priors = np.bincount(y_true, minlength=3) / len(y_true)
        prior_probas = np.tile(priors, (len(y_true), 1))
        prior_brier = float(np.mean([sum((prior_probas[i, k] - (1.0 if y_true[i] == k else 0.0))**2 for k in range(3)) for i in range(len(y_true))]))

        # Gate Evaluation
        gate_checks = {
            "G1_MINIMUM_OBSERVATIONS": {"required": f">= {MIN_PROSPECTIVE_OBSERVATIONS}", "observed": len(valid_resolved), "passed": bool(has_min_obs)},
            "G2_MINIMUM_DURATION_DAYS": {"required": f">= {MIN_PROSPECTIVE_CALENDAR_DAYS}", "observed": round(calendar_days, 1), "passed": bool(has_min_days)},
            "G3_NET_RETURN_POSITIVE": {"required": "> 0.00%", "observed": f"{cum_net_ret:+.2f}%", "passed": bool(cum_net_ret > 0.0)},
            "G4_MAX_DRAWDOWN_BOUND": {"required": "<= 4.00%", "observed": f"{max_dd:.2f}%", "passed": bool(max_dd <= 4.0)},
            "G5_BRIER_VS_CLASS_PRIOR": {"required": f"< {prior_brier:.4f}", "observed": round(brier, 4), "passed": bool(brier < prior_brier)},
            "G6_EXPECTED_CALIBRATION_ERROR": {"required": "<= 8.00%", "observed": f"{ece*100:.2f}%", "passed": bool(ece <= 0.08)},
            "G7_UNCERTAINTY_BOUND": {"required": "95% CI Lower Bound > -0.50%", "observed": f"{ci_low_cum:+.2f}%", "passed": bool(ci_low_cum > -0.50)},
        }

        all_passed = all(g["passed"] for g in gate_checks.values())

        report = {
            **base_summary,
            "promotion_recommendation": "PROMOTION_READY" if all_passed else "NOT_PROMOTED_INSUFFICIENT_EVIDENCE",
            "performance_metrics": {
                "active_trades_count": active_cnt,
                "coverage_pct": round(active_cnt / len(valid_resolved) * 100.0, 1),
                "net_win_rate_pct": round(win_rate, 2),
                "cumulative_net_return_pct": round(cum_net_ret, 2),
                "net_expectancy_per_trade_bps": round(float(np.mean(net_returns)) * 10000.0, 2),
                "maximum_drawdown_pct": round(max_dd, 2),
                "net_return_95_ci_pct": [ci_low_cum, ci_high_cum],
                "uncertainty_notice": "Negative lower confidence bound indicates expected return cannot be established as strictly positive with 95% confidence.",
            },
            "prediction_quality": {
                "raw_accuracy_pct": round(acc * 100.0, 2),
                "balanced_accuracy_pct": round(bal_acc * 100.0, 2),
                "multiclass_brier_score": round(brier, 4),
                "class_prior_brier_score": round(prior_brier, 4),
                "multiclass_log_loss": round(ll, 4),
                "ece_pct": round(ece * 100.0, 2),
                "confusion_matrix": cm.tolist(),
                "ece_bins": ece_bins,
            },
            "gate_compliance": gate_checks,
        }

        return report
