"""Trading Intelligence Platform - Operational CLI Model Governance & Promotion Auditor.

Evaluates candidate machine learning models against strict untouched out-of-sample criteria,
enforcing Controlled Self-Learning protocols without unverified automatic promotions.
"""

import argparse

import numpy as np
import pandas as pd

from learning.governance import ModelGovernor
from models.ml_classifier import LightGBMPredictor
from models.statistical import StatisticalBaselineModel


def main():
    parser = argparse.ArgumentParser(description="Audit and govern candidate trading models")
    parser.add_argument("--candidate-id", type=str, default="lgbm_alpha_v2", help="Candidate Model ID")
    parser.add_argument("--min-sharpe", type=float, default=1.0, help="Minimum OOS Sharpe required")
    parser.add_argument("--min-win-rate", type=float, default=50.0, help="Minimum OOS Win Rate (%)")
    args = parser.parse_args()

    print("\n========================================================")
    print("  CONTROLLED SELF-LEARNING - MODEL GOVERNANCE AUDIT")
    print("========================================================")
    print(f"Candidate Model ID:  {args.candidate_id}")
    print(f"Gate Target Sharpe:  >={args.min_sharpe:.2f}")
    print(f"Gate Target WinRate: >={args.min_win_rate:.1f}%")
    print("========================================================\n")

    # Generate synthetic historical train and out-of-sample test datasets
    np.random.seed(42)
    n_train = 300
    X_train = pd.DataFrame({
        "return_1": np.random.normal(0.001, 0.01, n_train),
        "log_return_5": np.random.normal(0.005, 0.02, n_train),
        "candle_range_pct": np.random.uniform(0.005, 0.02, n_train),
        "candle_body_ratio": np.random.uniform(0.1, 0.9, n_train),
        "rsi_14": np.random.uniform(25, 75, n_train),
        "macd_hist": np.random.normal(0, 0.5, n_train),
        "adx_14": np.random.uniform(15, 45, n_train),
        "atr_14": np.random.uniform(10, 50, n_train),
        "rvol_20": np.random.uniform(0.8, 2.5, n_train),
        "volatility_regime": np.random.randint(0, 4, n_train),
    })
    y_train = np.random.randint(0, 3, n_train)

    # Train candidate model
    print(f"1. Training candidate {args.candidate_id} on {n_train} feature rows...")
    candidate = LightGBMPredictor(model_id=args.candidate_id, n_estimators=30)
    candidate.fit(X_train, y_train)

    # Create baseline production model
    baseline = StatisticalBaselineModel(model_id="prod_stat_baseline")

    # Generate untouched Out-Of-Sample test dataset
    n_test = 200
    oos_returns = np.random.normal(0.0006, 0.007, n_test)
    oos_returns[30:80] = 0.004
    oos_closes = 65000.0 * np.exp(np.cumsum(oos_returns))
    oos_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-01", periods=n_test, freq="1h", tz="UTC"),
        "open": oos_closes * 0.999,
        "high": oos_closes * 1.004,
        "low": oos_closes * 0.996,
        "close": oos_closes,
        "volume": np.random.uniform(50, 200, n_test),
    })

    print(f"2. Running rigorous out-of-sample validation over {n_test} untouched bars...")
    governor = ModelGovernor(
        min_oos_sharpe=args.min_sharpe,
        min_oos_win_rate=args.min_win_rate,
        max_oos_drawdown=15.0,
    )
    verdict = governor.evaluate_candidate(
        candidate_model=candidate,
        oos_df=oos_df,
        production_model=baseline,
    )

    print("\n---------------- AUDIT VERDICT ----------------")
    print(f"Status:                    {verdict.status.value}")
    print(f"Reason:                    {verdict.reason}")
    print("Candidate OOS Metrics:")
    for k, v in verdict.candidate_metrics.items():
        print(f"  - {k}: {v}")
    print("Promotion Gate Checks:")
    for k, v in verdict.gate_checks.items():
        print(f"  - {k}: {'PASS [OK]' if v else 'FAIL [X]'}")
    print("-----------------------------------------------\n")


if __name__ == "__main__":
    main()
