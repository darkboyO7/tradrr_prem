"""Trading Intelligence Platform - Paper Trader & Execution Loop Runner.

Demonstrates end-to-end execution of the complete trading architecture:
1. Real-time / historical candle ingestion
2. Multi-feature extraction (price, trend, momentum, volatility, volume, structure)
3. 3D Regime classification (Trend, Volatility, Liquidity)
4. Signal evaluation and multi-evidence fusion
5. Pre-trade risk assessment and dynamic position sizing
6. Order routing to execution layer (SimulatedExecutionClient / PaperTradingEngine)
7. Dynamic Stop-Loss / Take-Profit order updates and position reconciliation
"""

import argparse
import asyncio

import numpy as np
import pandas as pd

from analysis.regime import MarketRegimeDetector
from core.database import close_db, init_db
from core.events import Candle
from core.logging import get_logger, setup_logging
from execution.reconciliation import PositionReconciler
from execution.simulated import SimulatedExecutionClient
from features.pipeline import FeaturePipeline
from models.base import SignalAction
from paper_trading.engine import PaperTradingEngine
from paper_trading.models import OrderSide, OrderType, PaperOrder
from risk.engine import RiskDecisionStatus, RiskEngine
from risk.limits import RiskLimits
from signals.fusion import SignalFusionEngine
from signals.technical import TechnicalSignalGenerator

logger = get_logger("paper_runner")


def generate_synthetic_candles(symbol: str, count: int = 100, base_price: float = 65000.0) -> list[Candle]:
    """Generate realistic synthetic OHLCV candles with micro-trends for simulation."""
    candles = []
    current_price = base_price
    t = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=count)

    for i in range(count):
        drift = 0.0003 * np.sin(i / 10.0)
        ret = np.random.normal(loc=drift, scale=0.004)
        open_p = current_price
        close_p = open_p * (1.0 + ret)
        high_p = max(open_p, close_p) * (1.0 + abs(np.random.normal(0, 0.002)))
        low_p = min(open_p, close_p) * (1.0 - abs(np.random.normal(0, 0.002)))
        vol = float(np.random.uniform(10.0, 50.0))

        c = Candle(
            symbol=symbol,
            timeframe="1m",
            timestamp=(t + pd.Timedelta(minutes=i)).to_pydatetime(),
            open=round(open_p, 2),
            high=round(high_p, 2),
            low=round(low_p, 2),
            close=round(close_p, 2),
            volume=round(vol, 4),
        )
        candles.append(c)
        current_price = close_p

    return candles


async def run_paper_trader(symbol: str, num_candles: int, delay_s: float) -> None:
    """Run paper trading loop processing streamed candles."""
    logger.info("Initializing Paper Trader with Full Architecture Stack...")

    # 1. Initialize Database
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database init warning (continuing with in-memory state): {e}")

    # 2. Components
    feature_pipeline = FeaturePipeline()
    regime_detector = MarketRegimeDetector(feature_pipeline=feature_pipeline)
    technical_generator = TechnicalSignalGenerator()
    fusion_engine = SignalFusionEngine()
    risk_limits = RiskLimits(max_drawdown_limit_pct=0.10, max_positions_total=3)
    risk_engine = RiskEngine(initial_capital=100000.0, limits=risk_limits)

    paper_engine = PaperTradingEngine(initial_cash=100000.0)
    exec_client = SimulatedExecutionClient(paper_engine=paper_engine, simulated_latency_ms=5.0)
    await exec_client.connect()
    reconciler = PositionReconciler()

    logger.info("Generating candle sequence...", symbol=symbol, count=num_candles)
    candle_stream = generate_synthetic_candles(symbol=symbol, count=num_candles)

    # Historical buffer for feature engine
    candle_history: list[Candle] = candle_stream[:60]
    stream_feed: list[Candle] = candle_stream[60:]

    logger.info(
        "Beginning Live Simulation Loop",
        initial_history=len(candle_history),
        incoming_ticks=len(stream_feed),
    )

    trade_count = 0

    for idx, candle in enumerate(stream_feed):
        candle_history.append(candle)
        exec_client.set_market_price(symbol, candle.close)

        # Build DataFrame for Feature Engine
        df_candles = pd.DataFrame([
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candle_history
        ])

        # Step A: Features
        features_df = feature_pipeline.compute_all_features(df_candles)
        last_row = features_df.iloc[-1]

        # Step B: Regime
        regime = regime_detector.classify_bar(last_row)

        # Step C: Signal Generation & Fusion
        tech_signals = technical_generator.evaluate_bar(last_row)
        atr_val = float(last_row.get("atr_14", candle.close * 0.01))
        fused_decision = fusion_engine.fuse(
            symbol=symbol,
            current_price=candle.close,
            atr=atr_val,
            regime=regime,
            technical_signals=tech_signals,
            timestamp=candle.timestamp,
        )

        # Step D: Pre-Trade Risk Evaluation
        current_equity = await exec_client.get_balance()
        current_positions = await exec_client.get_positions()
        pos_exposure = {p["symbol"]: p["quantity"] * p["current_price"] for p in current_positions}
        risk_engine.update_portfolio_state(current_equity=paper_engine.equity, open_positions=pos_exposure)

        # Check existing position for this symbol
        active_pos = next((p for p in current_positions if p["symbol"] == symbol), None)

        if fused_decision.action != SignalAction.NO_TRADE and not active_pos:
            risk_decision = risk_engine.evaluate_trade(signal=fused_decision)

            if risk_decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED) and risk_decision.approved_quantity > 0:
                # Route to Execution Client
                side = OrderSide.BUY if fused_decision.action == SignalAction.BUY else OrderSide.SELL
                order = PaperOrder(
                    symbol=symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=risk_decision.approved_quantity,
                    requested_price=candle.close,
                    stop_loss=risk_decision.stop_loss,
                    take_profit=risk_decision.take_profit,
                )

                fill = await exec_client.submit_order(order)
                trade_count += 1
                logger.info(
                    "Order Filled Successfully",
                    action=fused_decision.action.value,
                    price=fill.filled_price,
                    qty=fill.filled_quantity,
                    fee=fill.fee,
                    regime=regime.regime_code,
                )

        if delay_s > 0:
            await asyncio.sleep(delay_s)

    # Step E: Reconciliation Check
    remote_positions = await exec_client.get_positions()
    recon_report = reconciler.reconcile(
        local_positions=paper_engine.positions,
        remote_positions=remote_positions,
    )

    final_equity = paper_engine.equity
    pnl = final_equity - paper_engine.initial_cash

    print("\n" + "=" * 60)
    print("           PAPER TRADING SIMULATION SUMMARY           ")
    print("=" * 60)
    print(f"Symbol:                 {symbol}")
    print(f"Candles Processed:      {len(candle_stream)}")
    print(f"Initial Cash:           ${paper_engine.initial_cash:,.2f}")
    print(f"Final Equity:           ${final_equity:,.2f}")
    print(f"Net Realized/Unreal PnL:${pnl:+,.2f} ({(pnl / paper_engine.initial_cash) * 100:.2f}%)")
    print(f"Executed Trades:        {trade_count}")
    print(f"Open Positions:         {len(paper_engine.positions)}")
    print(f"Reconciliation Synced:  {recon_report.is_synchronized}")
    print("=" * 60 + "\n")

    await exec_client.disconnect()
    await close_db()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading Intelligence Platform - Paper Trader")
    parser.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol to trade")
    parser.add_argument("--candles", type=int, default=120, help="Number of simulated candles")
    parser.add_argument("--delay", type=float, default=0.0, help="Artificial delay per step in seconds")
    args = parser.parse_args()

    setup_logging(level="INFO")
    asyncio.run(run_paper_trader(symbol=args.symbol, num_candles=args.candles, delay_s=args.delay))


if __name__ == "__main__":
    main()
