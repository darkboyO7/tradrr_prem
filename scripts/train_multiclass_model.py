"""Trading Intelligence Platform — Balanced Multiclass Model Training Engine.

Trains a production-grade 3-class LightGBM predictor (0=SELL, 1=NEUTRAL, 2=BUY)
on authentic historical NSE & BSE market candles with purged embargo validation.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

from core.logging import get_logger, setup_logging
from data.ingestion.nse import NSEDataProvider
from data.ingestion.bse import BSEDataProvider
from features.pipeline import FeaturePipeline
from learning.accuracy_evaluator import (
    AccuracyAcceptanceState,
    InstitutionalModelEvaluator,
    InstitutionalWalkForwardEvaluator,
    ModelAccuracyEvaluationReport,
    WalkForwardAccuracyReport,
)
from models.ml_classifier import (
    CLASS_BUY,
    CLASS_NAMES,
    CLASS_NEUTRAL,
    CLASS_SELL,
    LightGBMPredictor,
    REQUIRED_NUM_CLASSES,
)

logger = get_logger("multiclass_trainer")

FEATURE_COLUMNS = [
    "return_1",
    "log_return_5",
    "candle_range_pct",
    "candle_body_ratio",
    "rsi_14",
    "macd_hist",
    "adx_14",
    "atr_14",
    "rvol_20",
    "volatility_regime",
]


def load_multi_instrument_candles(
    symbols: list[str],
    timeframe: str = "5m",
    limit_per_sym: int = 240,
) -> dict[str, pd.DataFrame]:
    """Retrieve authentic historical candles across target instruments."""
    nse = NSEDataProvider()
    bse = BSEDataProvider()
    datasets: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        clean_sym = sym.strip().upper()
        if clean_sym.startswith("BSE:"):
            candles = bse.fetch_historical_candles_sync(clean_sym, timeframe=timeframe, limit=limit_per_sym)
        else:
            candles = nse.fetch_historical_candles_sync(clean_sym, timeframe=timeframe, limit=limit_per_sym)

        if not candles or len(candles) < 30:
            logger.warning("Insufficient candles retrieved", symbol=clean_sym, count=len(candles) if candles else 0)
            continue

        df = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            }
            for c in candles
        ])
        df = df.sort_values("timestamp").reset_index(drop=True)
        datasets[clean_sym] = df
        logger.info("Loaded authentic candles", symbol=clean_sym, bars=len(df), timeframe=timeframe)

    return datasets


def build_labeled_features(
    datasets: dict[str, pd.DataFrame],
    lookahead_bars: int = 3,
    return_threshold: float = 0.0015,
    use_atr_deadband: bool = False,
    atr_multiplier: float = 0.25,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute features and forward 3-class target labels (0=SELL, 1=NEUTRAL, 2=BUY)."""
    pipe = FeaturePipeline()
    all_X_rows: list[pd.DataFrame] = []
    all_y_labels: list[np.ndarray] = []
    all_fwd_returns: list[np.ndarray] = []

    for sym, raw_df in datasets.items():
        if len(raw_df) <= lookahead_bars + 15:
            continue

        # 1. Compute full technical feature pipeline
        feat_df = pipe.compute_all_features(raw_df)

        # 2. Add volatility_regime ordinal if not present
        if "volatility_regime" not in feat_df.columns:
            if "atr_14" in feat_df.columns:
                atr_p50 = feat_df["atr_14"].median()
                feat_df["volatility_regime"] = np.where(feat_df["atr_14"] > atr_p50, 2, 1)
            else:
                feat_df["volatility_regime"] = 1

        # 3. Calculate forward lookahead return: R_{t+H} = (Close_{t+H} - Close_t) / Close_t
        closes = raw_df["close"].astype(float)
        fwd_close = closes.shift(-lookahead_bars)
        fwd_ret = ((fwd_close - closes) / closes).values

        # 4. Generate 3-class target labels
        if use_atr_deadband and "atr_14" in feat_df.columns:
            atr_vals = feat_df["atr_14"].values
            dyn_thresh = np.maximum(0.0005, atr_multiplier * (atr_vals / closes.values))
            y = np.where(
                fwd_ret > dyn_thresh,
                CLASS_BUY,
                np.where(fwd_ret < -dyn_thresh, CLASS_SELL, CLASS_NEUTRAL),
            )
        else:
            y = np.where(
                fwd_ret > return_threshold,
                CLASS_BUY,
                np.where(fwd_ret < -return_threshold, CLASS_SELL, CLASS_NEUTRAL),
            )

        # 5. Drop the tail lookahead bars where forward return is NaN
        valid_mask = ~np.isnan(fwd_ret)
        X_sub = feat_df.loc[valid_mask, [c for c in FEATURE_COLUMNS if c in feat_df.columns]].copy()
        y_sub = y[valid_mask]
        ret_sub = fwd_ret[valid_mask]

        all_X_rows.append(X_sub)
        all_y_labels.append(y_sub)
        all_fwd_returns.append(ret_sub)

    if not all_X_rows:
        return pd.DataFrame(), np.array([]), np.array([])

    X_total = pd.concat(all_X_rows, ignore_index=True).fillna(0.0)
    y_total = np.concatenate(all_y_labels)
    ret_total = np.concatenate(all_fwd_returns)

    return X_total, y_total, ret_total


def compute_multiclass_brier(probas: np.ndarray, y_true: np.ndarray) -> float:
    """Compute standard multiclass Brier score (mean squared error of probability vector)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    y_onehot = np.zeros((n, REQUIRED_NUM_CLASSES))
    for i, c in enumerate(y_true):
        if 0 <= c < REQUIRED_NUM_CLASSES:
            y_onehot[i, int(c)] = 1.0
    return float(np.mean(np.sum((probas - y_onehot) ** 2, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Institutional 3-Class LightGBM Training Engine")
    parser.add_argument(
        "--symbols",
        type=str,
        default="NSE:TCS,NSE:RELIANCE,NSE:INFY,NSE:IDEA,NSE:GROWW,NSE:TATAGOLD,NSE:TATSILV,NSE:IDBI,NSE:RTNINDIA",
        help="Comma-separated ticker list",
    )
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle resolution (5m, 15m, 1h)")
    parser.add_argument("--lookahead", type=int, default=3, help="Forecast horizon in bars")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0015,
        help="Price threshold for Neutral class band",
    )
    parser.add_argument(
        "--use-atr-deadband",
        action="store_true",
        help="Use dynamic ATR-proportional neutral deadband",
    )
    parser.add_argument(
        "--atr-multiplier",
        type=float,
        default=0.25,
        help="Multiplier for dynamic ATR deadband",
    )
    parser.add_argument(
        "--run-walk-forward",
        action="store_true",
        default=True,
        help="Run expanding-window walk-forward validation across sequential regimes",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train / OOS partition split")
    parser.add_argument("--embargo-bars", type=int, default=3, help="Purged embargo gap bars")
    parser.add_argument("--model-id", type=str, default="lgbm_multiclass_candidate", help="Model identifier")
    args = parser.parse_args()

    setup_logging(level="INFO")

    print("\n" + "=" * 76)
    print("  INSTITUTIONAL 3-CLASS LIGHTGBM MULTICLASS TRAINING ENGINE")
    print("=" * 76)
    print(f"Target Symbols:         {args.symbols}")
    print(f"Candle Timeframe:       {args.timeframe}")
    print(f"Forecast Horizon:       {args.lookahead} bars ({args.lookahead * 5}m)")
    if args.use_atr_deadband:
        print(f"Deadband Mode:          Dynamic ATR Deadband ({args.atr_multiplier:.2f} * ATR_14 / Close)")
    else:
        print(f"Neutral Threshold:      +/-{args.threshold * 100:.3f}%")
    print(f"Train / OOS Split:      {args.train_ratio * 100:.0f}% / {(1.0 - args.train_ratio) * 100:.0f}% (Embargo: {args.embargo_bars} bars)")
    print("=" * 76 + "\n")

    symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # 1. Ingest authentic candles
    datasets = load_multi_instrument_candles(symbols_list, timeframe=args.timeframe, limit_per_sym=240)
    if not datasets:
        print("[ERROR] No authentic market candles could be retrieved.")
        sys.exit(1)

    # 2. Extract features & 3-class target labels
    X, y, fwd_returns = build_labeled_features(
        datasets,
        lookahead_bars=args.lookahead,
        return_threshold=args.threshold,
        use_atr_deadband=args.use_atr_deadband,
        atr_multiplier=args.atr_multiplier,
    )

    n_samples = len(X)
    print(f"Dataset assembled: {n_samples} total observation rows across {len(datasets)} instruments.")

    # Display class counts
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("\nTarget Class Distribution:")
    for c_idx in (CLASS_SELL, CLASS_NEUTRAL, CLASS_BUY):
        c_count = class_dist.get(c_idx, 0)
        c_pct = (c_count / n_samples) * 100.0 if n_samples > 0 else 0.0
        print(f"  Class {c_idx} ({CLASS_NAMES[c_idx]:<7}): {c_count:4d} samples ({c_pct:5.1f}%)")

    if len(class_dist) < REQUIRED_NUM_CLASSES:
        print(f"[FAIL] Missing classes: found {len(class_dist)}/3. Cannot satisfy strict multiclass contract.")
        sys.exit(1)

    # 3. Chronological Purged Split with Embargo
    split_idx = int(n_samples * args.train_ratio)
    oos_start = min(n_samples - 4, split_idx + args.embargo_bars)

    X_train = X.iloc[:split_idx].copy()
    y_train = y[:split_idx]

    X_oos = X.iloc[oos_start:].copy()
    y_oos = y[oos_start:]
    ret_oos = fwd_returns[oos_start:]

    print(f"\nPartitions: Train={len(X_train)} bars, Embargo={oos_start - split_idx} bars, OOS={len(X_oos)} bars.")

    # 4. Train LightGBMPredictor with strict multiclass contract
    ver_str = f"v{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    candidate = LightGBMPredictor(
        model_id=args.model_id,
        version=ver_str,
        feature_names=list(X_train.columns),
        n_estimators=45,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=8,
        min_child_samples=5,
    )

    print(f"\nFitting LightGBMPredictor with strict_multiclass=True...")
    candidate.fit(X_train, y_train, strict_multiclass=True)

    print(f"Model fitted successfully!")
    print(f"  - Fitted classes:       {candidate.fitted_classes_} (All 3 classes verified)")
    print(f"  - Is Fully Supported:   {candidate.is_fully_supported}")
    print(f"  - Feature count:        {len(candidate.metadata.feature_names)}")

    # 5. Out-of-Sample Performance Evaluation (Section 8)
    oos_probas = candidate.predict_proba(X_oos)

    evaluator = InstitutionalModelEvaluator(
        target_accuracy=0.95,
        min_sample_size=100,
        min_active_trades=35,
        confidence_level=0.95,
    )
    eval_report = evaluator.evaluate(
        model_id=args.model_id,
        version=ver_str,
        y_true=y_oos,
        probas=oos_probas,
        fwd_returns=ret_oos,
        lookahead_bars=args.lookahead,
        embargo_bars=args.embargo_bars,
        forecast_horizon_label=f"{args.lookahead * 5}m",
        target_description=f"Forward {args.lookahead}-Bar Return 3-Class Target [0=SELL, 1=NEUTRAL, 2=BUY]",
    )

    print("\n" + "=" * 76)
    print("  INSTITUTIONAL OUT-OF-SAMPLE ACCURACY AUDIT (Section 8)")
    print("=" * 76)
    print(f"Acceptance State:       [{eval_report.acceptance_state.value}]")
    print(f"OOS Sample Size:        {eval_report.sample_size} observations")
    print(f"Overall Accuracy:       {eval_report.overall_accuracy_pct:.2f}% (Hurdle: 95.0%)")
    print(
        f"Wilson 95% Score CI:    [{eval_report.wilson_ci_lower_pct:.2f}%, {eval_report.wilson_ci_upper_pct:.2f}%] (Lower bound must >= 95.0%)"
    )
    print(f"Multiclass Brier Score: {eval_report.brier_score:.4f} (Uniform Baseline: 0.6667)")
    print(f"Expected Calib. Error:  {eval_report.expected_calibration_error:.4f}")
    print(
        f"Coverage / Abstention:  Coverage={eval_report.coverage_rate_pct:.1f}%, Abstention={eval_report.abstention_rate_pct:.1f}%"
    )
    print(f"Active Signals Count:   {eval_report.active_sample_count}/{eval_report.sample_size}")
    print(f"Active Trade Win Rate:  {eval_report.active_trade_win_rate_pct:.2f}%")
    print(f"Active Net Expectancy:  {eval_report.active_trade_expectancy_pct:+.4f}%")
    print("\n3x3 Confusion Matrix (Rows=True, Cols=Predicted):")
    print("              Pred: SELL   Pred: NEUTRAL   Pred: BUY")
    cm = eval_report.confusion_matrix
    print(f"  True SELL     : {cm[0][0]:8d}        {cm[0][1]:8d}    {cm[0][2]:8d}")
    print(f"  True NEUTRAL  : {cm[1][0]:8d}        {cm[1][1]:8d}    {cm[1][2]:8d}")
    print(f"  True BUY      : {cm[2][0]:8d}        {cm[2][1]:8d}    {cm[2][2]:8d}")

    print("\nPer-Class Breakdown:")
    for c_name in ("SELL", "NEUTRAL", "BUY"):
        pm = eval_report.per_class_metrics[c_name]
        print(
            f"  Class {c_name:<7}: Support={pm.support:4d} | Precision={pm.precision * 100:5.1f}% | Recall={pm.recall * 100:5.1f}% | F1={pm.f1_score:5.3f}"
        )

    if eval_report.rejection_reasons:
        print("\nAudit Caveats / Hurdle Gaps:")
        for reason in eval_report.rejection_reasons:
            print(f"  - {reason}")
    print("=" * 76)

    # 6. Chronological Walk-Forward Cross-Validation (Expanding Window)
    wf_report = None
    if args.run_walk_forward and len(X) >= 100:
        print("\n" + "=" * 76)
        print("  CHRONOLOGICAL WALK-FORWARD CROSS-VALIDATION (4 FOLDS)")
        print("=" * 76)
        wf_evaluator = InstitutionalWalkForwardEvaluator(
            n_splits=4,
            initial_train_ratio=0.50,
            embargo_bars=args.embargo_bars,
            lookahead_bars=args.lookahead,
            target_accuracy=0.95,
            min_sample_size=30,
        )

        def make_fold_candidate():
            return LightGBMPredictor(
                model_id=args.model_id,
                version="wf_fold",
                feature_names=list(X.columns),
                n_estimators=45,
                learning_rate=0.04,
                max_depth=3,
                num_leaves=8,
                min_child_samples=5,
            )

        try:
            wf_report = wf_evaluator.run_walk_forward(
                model_factory=make_fold_candidate,
                X=X,
                y=y,
                fwd_returns=fwd_returns,
                model_id=args.model_id,
                horizon_label=f"{args.lookahead * 5}m",
            )
            print(f"Completed {wf_report.n_folds} chronological folds across {wf_report.total_oos_samples} total OOS bars:")
            for f in wf_report.folds:
                r = f.eval_report
                print(
                    f"  Fold {f.fold_index}: Train={f.train_bars:4d} bars | OOS={f.oos_bars:3d} bars | "
                    f"Acc={r.overall_accuracy_pct:5.1f}% | WinRate={r.active_trade_win_rate_pct:5.1f}% | Exp={r.active_trade_expectancy_pct:+.4f}%"
                )
            print(f"\nMean Fold Accuracy:     {wf_report.mean_fold_accuracy_pct:.2f}% (+/- {wf_report.std_fold_accuracy_pct:.2f}%)")
            print(f"Mean Fold Win Rate:     {wf_report.mean_fold_win_rate_pct:.2f}%")
            print(f"Mean Fold Expectancy:   {wf_report.mean_fold_expectancy_pct:+.4f}%")
            print(
                f"Aggregate OOS Acc:      {wf_report.aggregate_report.overall_accuracy_pct:.2f}% "
                f"(Wilson 95% CI: [{wf_report.aggregate_report.wilson_ci_lower_pct:.2f}%, {wf_report.aggregate_report.wilson_ci_upper_pct:.2f}%])"
            )
            print("=" * 76)
        except Exception as exc:
            logger.warning("Walk-forward cross-validation skipped or failed", error=str(exc))

    # 7. Save Candidate Model to Staging Store
    out_dir = Path("models_store") / args.model_id / ver_str
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metadata.json"
    audit_report_path = out_dir / "accuracy_evaluation_report.json"
    wf_report_path = out_dir / "walk_forward_accuracy_report.json"

    joblib.dump(candidate.classifier, model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metadata_payload = {
        "model_id": args.model_id,
        "version": ver_str,
        "created_at": datetime.now(UTC).isoformat(),
        "is_production_ready": False,  # Governed: must satisfy 60-day prospective evidence
        "multiclass_classes": candidate.fitted_classes_,
        "class_names": list(CLASS_NAMES),
        "is_fully_supported": candidate.is_fully_supported,
        "alpha_dirichlet_smoothing": 0.15,
        "feature_names": candidate.metadata.feature_names,
        "model_sha256": model_sha256,
        "train_samples": len(X_train),
        "oos_samples": len(X_oos),
        "oos_metrics": {
            "accuracy_pct": eval_report.overall_accuracy_pct,
            "wilson_ci_lower_pct": eval_report.wilson_ci_lower_pct,
            "wilson_ci_upper_pct": eval_report.wilson_ci_upper_pct,
            "acceptance_state": eval_report.acceptance_state.value,
            "multiclass_brier_score": eval_report.brier_score,
            "expected_calibration_error": eval_report.expected_calibration_error,
            "active_win_rate_pct": eval_report.active_trade_win_rate_pct,
            "active_expectancy_pct": eval_report.active_trade_expectancy_pct,
        },
        "walk_forward_metrics": (
            {
                "n_folds": wf_report.n_folds,
                "mean_fold_accuracy_pct": wf_report.mean_fold_accuracy_pct,
                "std_fold_accuracy_pct": wf_report.std_fold_accuracy_pct,
                "mean_fold_win_rate_pct": wf_report.mean_fold_win_rate_pct,
                "mean_fold_expectancy_pct": wf_report.mean_fold_expectancy_pct,
                "aggregate_oos_accuracy_pct": wf_report.aggregate_report.overall_accuracy_pct,
                "aggregate_wilson_ci_lower_pct": wf_report.aggregate_report.wilson_ci_lower_pct,
                "aggregate_wilson_ci_upper_pct": wf_report.aggregate_report.wilson_ci_upper_pct,
            }
            if wf_report
            else None
        ),
        "governance_note": (
            "Model evaluated under Section 8 Institutional Accuracy & Evaluation Standard. "
            f"Acceptance State: {eval_report.acceptance_state.value}."
        ),
    }

    meta_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    meta_sha256 = hashlib.sha256(meta_path.read_bytes()).hexdigest()

    audit_report_path.write_text(json.dumps(eval_report.to_dict(), indent=2), encoding="utf-8")
    if wf_report:
        wf_report_path.write_text(json.dumps(wf_report.to_dict(), indent=2), encoding="utf-8")

    print(f"\nModel Staged Successfully:")
    print(f"  Artifact Path:  {out_dir}")
    print(f"  Model Hash:     {model_sha256[:16]}...")
    print(f"  Metadata Hash:  {meta_sha256[:16]}...")
    print(f"  Readiness:      is_production_ready = False (Governed Staging)")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
