"""Trading Intelligence Platform - Paper-Trading Background Worker Runner.

Provides a standalone daemon/service entry point for the paper-trading worker.
Handles:
- Rotating file logging to logs/paper_worker.log and stdout.
- OS process signal trapping (SIGINT, SIGTERM) for graceful resource and lock release.
- Continuous loop execution without GUI dependencies.

Usage:
    python scripts/run_paper_worker.py
    python scripts/run_paper_worker.py --poll-interval 60 --store-dir data/paper_trading
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_trading.worker import PaperTradingWorker, WorkerState


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "paper_worker.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Formatter
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] [PID:%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler (10 MB x 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    return logging.getLogger("paper_trading.runner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prospective Multi-Market Paper-Trading Background Worker")
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Poll interval in seconds (default: 60s)")
    parser.add_argument("--store-dir", type=str, default="data/paper_trading", help="Paper store directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--markets", type=str, default="NSE,BSE,QUOTEX", help="Comma-separated target markets (NSE, BSE, QUOTEX)")
    parser.add_argument("--instruments", type=str, default=None, help="Comma-separated instrument symbols (e.g. NSE:TCS,BSE:RELIANCE,QUOTEX:BTCUSD_QX)")
    parser.add_argument("--research-mode", action="store_true", default=True, help="Enable research mode for Quotex underlying crypto proxies")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / args.log_dir
    logger = setup_logging(log_dir)

    target_markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    configured_instruments = [sym.strip() for sym in args.instruments.split(",") if sym.strip()] if args.instruments else None

    logger.info("=================================================================")
    logger.info("Initializing Multi-Market Paper-Trading Background Worker...")
    logger.info(f"Configured Markets: {target_markets}")
    logger.info(f"Store Directory: {args.store_dir}")
    logger.info(f"Poll Interval: {args.poll_interval}s")
    logger.info("Governance Policy: POL-GOV-2026-001 (FROZEN)")
    logger.info("Trading Mode: SIMULATED ONLY - REAL TRADING PERMANENTLY DISABLED")
    logger.info("Quotex Economics: Underlying spot crypto proxy only (Research Mode)")
    logger.info("=================================================================")

    worker = PaperTradingWorker(
        store_dir=PROJECT_ROOT / args.store_dir,
        instruments=configured_instruments,
        poll_interval_seconds=args.poll_interval,
    )

    # Signal handlers
    def _shutdown_handler(signum, frame):
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.info(f"Received shutdown signal {sig_name}. Initiating graceful worker shutdown...")
        worker.stop(timeout=10.0)
        logger.info("Shutdown sequence complete. Exiting runner.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # Start worker
    success = worker.start()
    if not success:
        logger.error(f"Failed to start PaperTradingWorker. Reason: {worker._last_error}")
        return 1

    logger.info(f"Worker is RUNNING (PID: {os.getpid()}). Press Ctrl+C to terminate.")

    try:
        while worker.state in (WorkerState.RUNNING, WorkerState.DEGRADED):
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        worker.stop(timeout=10.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
