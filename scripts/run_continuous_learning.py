"""Trading Intelligence Platform - Continuous Learning & Governance Runner.

Automates the background monitoring, chronological retraining, out-of-sample
governance validation, and model promotion cycle.
"""

import argparse

import numpy as np
import pandas as pd

from core.logging import get_logger, setup_logging
from learning.loop import ContinuousLearningCoordinator
from models.ml_classifier import LightGBMPredictor

logger = get_logger("learning_runner")


def generate_market_data(count: int = 300, base_price: float = 65000.0) -> pd.DataFrame:
    """Generate chronological synthetic OHLCV candle series for training & OOS testing."""
    t_start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=count)
    timestamps = [t_start + pd.Timedelta(minutes=i) for i in range(count)]

    # Generate synthetic price path with autocorrelation and regimes
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.004, count)
    prices = base_price * np.exp(np.cumsum(returns))

    opens = prices * (1.0 + np.random.normal(0, 0.001, count))
    closes = prices
    highs = np.maximum(opens, closes) * (1.0 + np.abs(np.random.normal(0, 0.002, count)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(np.random.normal(0, 0.002, count)))
    volumes = np.random.uniform(10.0, 50.0, count)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Intelligence Platform - Continuous Learning Daemon")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Target symbol")
    parser.add_argument("--candles", type=int, default=300, help="Number of historical candles to analyze")
    parser.add_argument("--force", action="store_true", help="Force model retraining regardless of degradation status")
    args = parser.parse_args()

    setup_logging(level="INFO")
    logger.info("Initializing Autonomous Continuous Learning Cycle...", symbol=args.symbol, candles=args.candles)

    # 1. Generate / load historical candles
    df = generate_market_data(count=args.candles)

    # 2. Coordinator
    coordinator = ContinuousLearningCoordinator()

    # 3. Setup baseline production model
    initial_prod_model = LightGBMPredictor(
        model_id="prod_baseline_v1",
        version="v1.0.0",
        n_estimators=30,
        learning_rate=0.05,
    )
    coordinator.set_production_model(initial_prod_model)

    # 4. Execute Governance Cycle
    result = coordinator.execute_governance_cycle(
        historical_candles_df=df,
        force_retrain=args.force or True,
    )

    print("\n" + "=" * 65)
    print("           CONTINUOUS LEARNING & GOVERNANCE SUMMARY           ")
    print("=" * 65)
    print(f"Triggered:              {result.triggered}")
    print(f"Trigger Reason:         {result.trigger_reason}")
    print(f"Active Model ID:        {result.active_model_id}")
    print(f"Candidate Model ID:     {result.candidate_model_id}")
    print(f"Promoted to Production: {result.promoted_to_production}")

    if result.promotion_verdict:
        v = result.promotion_verdict
        print("-" * 65)
        print(f"Promotion Verdict:      {v.get('status')}")
        print(f"Verdict Rationale:      {v.get('reason')}")
        cand_m = v.get("candidate_metrics", {})
        print(f"OOS Sharpe Ratio:       {cand_m.get('sharpe', 0.0):.2f}")
        print(f"OOS Win Rate:           {cand_m.get('win_rate', 0.0):.1f}%")
        print(f"OOS Max Drawdown:       {cand_m.get('max_drawdown', 0.0):.1f}%")
        print(f"OOS Profit Factor:      {cand_m.get('profit_factor', 0.0):.2f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
