"""Trading Intelligence Platform - Unified Master CLI Controller.

Single point of command-line operations for running ingestion, paper trading,
event-driven backtesting, model governance, continuous self-learning, and the REST/Dashboard server.
"""

import argparse
import asyncio
import sys

from core.config import settings
from core.database import close_db, init_db
from core.logging import get_logger, setup_logging
from monitoring.health import check_system_health

logger = get_logger("cli")


async def run_status_command() -> int:
    """Check and display comprehensive health and subsystem status."""
    print("\n" + "=" * 65)
    print("       TRADING INTELLIGENCE PLATFORM - SYSTEM DIAGNOSTICS      ")
    print("=" * 65)
    try:
        try:
            await init_db()
        except Exception:
            pass
        health = await check_system_health()
        print(f"Overall Status:         {health['status'].upper()}")
        print(f"Uptime:                 {health['uptime_seconds']}s")
        print(f"Operating System:       {health['environment']['os']} {health['environment']['os_release']}")
        print(f"Python Version:         {health['environment']['python_version']}")
        print("-" * 65)
        print("Subsystems:")
        for comp, detail in health["components"].items():
            st = detail.get("status", "unknown") if isinstance(detail, dict) else str(detail)
            print(f"  - {comp.capitalize():<20}: {str(st).upper()}")
        print("=" * 65 + "\n")
        await close_db()
        return 0
    except Exception as e:
        print(f"Diagnostics Error: {e}")
        return 1


def main(args: list[str] | None = None) -> int:
    """Master CLI router."""
    parser = argparse.ArgumentParser(
        prog="trading-cli",
        description="Institutional Trading Intelligence Platform - Master Controller",
    )
    subparsers = parser.add_subparsers(dest="command", help="Platform operational command to execute")

    # 1. Status Command
    subparsers.add_parser("status", help="Run holistic system health diagnostics")

    # 2. Ingest Command
    p_ingest = subparsers.add_parser("ingest", help="Start live market data ingestion pipeline")
    p_ingest.add_argument("--provider", choices=["coinbase", "binance"], default="coinbase", help="Market data venue")
    p_ingest.add_argument("--symbols", type=str, default="BTC-USD,ETH-USD", help="Comma-separated symbols")

    # 3. Paper Trading Command
    p_paper = subparsers.add_parser("paper", help="Run live paper trading and simulated execution")
    p_paper.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    p_paper.add_argument("--candles", type=int, default=100, help="Number of simulated candles")
    p_paper.add_argument("--delay", type=float, default=0.0, help="Delay between ticks (seconds)")

    # 4. Backtest Command
    p_backtest = subparsers.add_parser("backtest", help="Run event-driven walk-forward backtest")
    p_backtest.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    p_backtest.add_argument("--candles", type=int, default=150, help="Historical candle count")
    p_backtest.add_argument("--initial-capital", type=float, default=100000.0, help="Starting balance ($)")

    # 5. Model Governance Command
    p_govern = subparsers.add_parser("govern", help="Run strict out-of-sample model governance audit")
    p_govern.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    p_govern.add_argument("--candles", type=int, default=150, help="Candle sample size")

    # 6. Continuous Learning Command
    p_learn = subparsers.add_parser("learn", help="Run autonomous continuous retraining loop")
    p_learn.add_argument("--symbol", type=str, default="BTC-USD", help="Ticker symbol")
    p_learn.add_argument("--candles", type=int, default=150, help="Training + OOS candle size")
    p_learn.add_argument("--force", action="store_true", help="Force retraining cycle")

    # 7. Serve Command
    p_serve = subparsers.add_parser("serve", help="Start FastAPI REST, Prometheus & Web Dashboard server")
    p_serve.add_argument("--host", type=str, default=settings.APP_HOST, help="Bind interface")
    p_serve.add_argument("--port", type=int, default=settings.APP_PORT, help="Port to listen on")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parsed = parser.parse_args(args)

    setup_logging(level=settings.LOG_LEVEL, json_format=settings.LOG_JSON)

    if not parsed.command or parsed.command == "status":
        return asyncio.run(run_status_command())

    elif parsed.command == "paper":
        from scripts.run_paper_trader import run_paper_trader
        asyncio.run(run_paper_trader(symbol=parsed.symbol, num_candles=parsed.candles, delay_s=parsed.delay))
        return 0

    elif parsed.command == "backtest":
        from scripts.run_backtest import run_backtest_cli
        run_backtest_cli(symbol=parsed.symbol, candles_count=parsed.candles, initial_capital=parsed.initial_capital)
        return 0

    elif parsed.command == "govern":
        from scripts.run_model_governance import run_governance_cli
        run_governance_cli(symbol=parsed.symbol, candles_count=parsed.candles)
        return 0

    elif parsed.command == "learn":
        from learning.loop import ContinuousLearningCoordinator
        from models.ml_classifier import LightGBMPredictor
        from scripts.run_continuous_learning import generate_market_data

        df = generate_market_data(count=parsed.candles)
        coordinator = ContinuousLearningCoordinator()
        initial_prod = LightGBMPredictor(model_id="prod_baseline_v1", version="v1.0.0", n_estimators=30)
        coordinator.set_production_model(initial_prod)
        res = coordinator.execute_governance_cycle(historical_candles_df=df, force_retrain=parsed.force or True)
        print(f"Autonomous Learning Verdict: Promoted={res.promoted_to_production}, Triggered={res.triggered}")
        return 0

    elif parsed.command == "serve":
        import uvicorn
        print(f"Starting Trading Intelligence API & Dashboard at http://{parsed.host}:{parsed.port}")
        uvicorn.run("api.main:app", host=parsed.host, port=parsed.port, reload=parsed.reload)
        return 0

    elif parsed.command == "ingest":
        from scripts.run_live_ingestion import main as run_ingest
        asyncio.run(run_ingest())
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
