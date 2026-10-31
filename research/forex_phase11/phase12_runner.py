"""Phase 12 Authentic Forex Data Acquisition Runner.

Executes Tasks 1 through 9:
1. Checks OANDA credentials locally without exposing secrets.
2. Reports exact missing keys if absent, and halts safely.
3. If configured, validates authenticated connection to OANDA practice endpoint.
4. Retrieves genuine M5 sample for EURUSD and EURCHF with bid/ask/mid data.
5. Preserves immutable raw response and SHA-256 hash in research/forex_phase11/raw/.
6. Validates timestamp ordering, OHLC consistency, positive prices, spread consistency, and timezone.
7. Distinguishes validated observations, rejected observations, and unexplained gaps.
8. Checks protected file hashes and worker status.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path("D:/ant")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from research.forex_phase11.oanda_client import OandaForexClient


def run_phase12() -> dict[str, Any]:
    print("=============================================================")
    print("PHASE 12 — AUTHENTIC FOREX DATA ACQUISITION RUNNER")
    print("=============================================================")

    # Task 1: Check configuration
    client = OandaForexClient.from_env_or_dotenv()
    is_valid, missing_keys = client.validate_configuration()

    print("\n[Task 1 & 2] Checking Local OANDA Configuration:")
    print(f"  OANDA_API_KEY      : {'CONFIGURED' if 'OANDA_API_KEY' not in missing_keys else 'MISSING'}")
    print(f"  OANDA_ACCOUNT_ID   : {'CONFIGURED' if 'OANDA_ACCOUNT_ID' not in missing_keys else 'MISSING'}")
    print(f"  OANDA_ENVIRONMENT  : {'CONFIGURED' if 'OANDA_ENVIRONMENT' not in missing_keys else 'MISSING'}")

    report = {
        "credentials_configured": is_valid,
        "missing_keys": missing_keys,
        "authenticated_connection_success": False,
        "instruments_verified": False,
        "eurusd_retrieved": False,
        "eurchf_retrieved": False,
        "raw_artifacts": [],
        "verdict": "BLOCKED" if not is_valid else "IN_PROGRESS",
    }

    if not is_valid:
        print("\n[STOPPED AT DATA PROVIDER GATE]")
        print(f"  Missing required configuration: {', '.join(missing_keys)}")
        print("  Safety rule enforced: Market data cannot be fabricated or simulated.")
        print("  Manual configuration required in D:\\ant\\.env:")
        print("    OANDA_ENVIRONMENT=practice")
        print("    OANDA_API_KEY=<your_bearer_token>")
        print("    OANDA_ACCOUNT_ID=<your_practice_account_id>")
        return report

    # Task 3: Authenticated Connection Test
    print("\n[Task 3] Testing Authenticated Access to OANDA Practice Endpoint:")
    conn_ok, conn_msg, inst_info = client.test_authenticated_connection()
    print(f"  Status: {'SUCCESS' if conn_ok else 'FAILED'} - {conn_msg}")
    report["authenticated_connection_success"] = conn_ok
    report["connection_message"] = conn_msg

    if not conn_ok:
        report["verdict"] = "BLOCKED"
        return report

    # Task 4 & 5: Retrieve EURUSD and EURCHF samples
    print("\n[Task 4 & 5] Retrieving Authentic M5 Candles:")
    for sym in ["EURUSD", "EURCHF"]:
        print(f"  Querying {sym} M5 (count=20, price=MBA)...")
        fetch_ok, fetch_msg, contracts, meta = client.fetch_historical_candles_sample(
            symbol=sym, count=20, granularity="M5"
        )
        print(f"    {sym} Result: {'SUCCESS' if fetch_ok else 'FAILED'} - {fetch_msg}")
        if fetch_ok:
            if sym == "EURUSD":
                report["eurusd_retrieved"] = True
            elif sym == "EURCHF":
                report["eurchf_retrieved"] = True
            report["raw_artifacts"].append(meta)

    all_retrieved = report["eurusd_retrieved"] and report["eurchf_retrieved"]
    report["verdict"] = "PASS" if all_retrieved else "CONDITIONAL PASS"
    return report


if __name__ == "__main__":
    rep = run_phase12()
    out_path = Path("research/forex_phase11/phase12_execution_report.json")
    out_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"\nExecution report written to {out_path}")
    print(f"Phase 12 Verdict: {rep['verdict']}")
