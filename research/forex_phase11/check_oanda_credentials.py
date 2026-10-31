"""Credential Presence Checker for Phase 12.

Checks presence of OANDA configuration keys in os.environ and .env.
STRICT SAFETY RULE: Never prints or exposes secret values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

def check_credentials() -> dict[str, dict[str, bool]]:
    env_keys = set(os.environ.keys())

    dot_env = Path(".env")
    dot_env_keys = set()
    if dot_env.exists():
        for line in dot_env.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k = line.split("=", 1)[0].strip()
                dot_env_keys.add(k)

    required = ["OANDA_API_KEY", "OANDA_ACCOUNT_ID", "OANDA_ENVIRONMENT"]
    status = {}
    for req in required:
        in_env = req in env_keys
        in_dot = req in dot_env_keys
        status[req] = {
            "in_os_environ": in_env,
            "in_dotenv_file": in_dot,
            "is_configured": in_env or in_dot,
        }
    return status

if __name__ == "__main__":
    res = check_credentials()
    all_configured = all(v["is_configured"] for v in res.values())
    missing = [k for k, v in res.items() if not v["is_configured"]]

    print("=== OANDA CREDENTIAL STATUS ===")
    for k, v in res.items():
        state = "PRESENT" if v["is_configured"] else "MISSING"
        print(f"  {k:<20}: {state} (env: {v['in_os_environ']}, .env: {v['in_dotenv_file']})")

    print("================================")
    print(f"All configured: {all_configured}")
    if missing:
        print(f"Missing configuration keys: {', '.join(missing)}")
    else:
        print("All required configuration keys are present.")
