"""Command-line operational diagnostics utility for prospective paper-trading.

Usage:
    python scripts/paper_diagnostics.py --status
    python scripts/paper_diagnostics.py --health-report
    python scripts/paper_diagnostics.py --evaluate
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_trading.evaluator import PaperTradingEvaluator
from paper_trading.health_report import generate_daily_health_report, render_health_report_markdown
from paper_trading.worker import PaperTradingWorker


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-Trading Operational Diagnostics & Health Reporter")
    parser.add_argument("--status", action="store_true", help="Print real-time operational status (JSON)")
    parser.add_argument("--health-report", action="store_true", help="Print daily health report (Markdown)")
    parser.add_argument("--health-report-json", action="store_true", help="Print daily health report (JSON)")
    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation against POL-GOV-2026-001")
    parser.add_argument("--store-dir", type=str, default="data/paper_trading", help="Custom paper store directory")

    args = parser.parse_args()

    # Default to status if no flag provided
    if not (args.status or args.health_report or args.health_report_json or args.evaluate):
        args.status = True

    store_path = Path(args.store_dir)

    if args.status:
        status = PaperTradingWorker.get_operational_status(store_dir=store_path)
        print(json.dumps(status, indent=2))
        return 0

    if args.health_report:
        report = generate_daily_health_report(store_dir=store_path)
        print(render_health_report_markdown(report))
        return 0

    if args.health_report_json:
        report = generate_daily_health_report(store_dir=store_path)
        print(json.dumps(report, indent=2))
        return 0

    if args.evaluate:
        evaluator = PaperTradingEvaluator(store_dir=store_path)
        report = evaluator.evaluate()
        print(json.dumps(report, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
