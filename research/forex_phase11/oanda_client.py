"""Authentic OANDA v20 REST Client for Isolated Forex Research.

Adheres to:
- POL-GOV-2026-001 (Research-only isolation, zero production ledger writes)
- FX-DATA-CONTRACT-v1.0.0 (Strict price basis, UTC timestamps, discrete bid/ask)

Security Guarantees:
- Never prints, logs, or writes API tokens to disk or console.
- Credentials loaded from os.environ or local .env only.
- Strict fail-closed error handling when credentials are missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.contracts.forex_contract import (
    APPROVED_FX_SYMBOLS,
    PERMANENT_BLOCKLIST,
    PriceBasis,
    ForexCandleContractV1,
    validate_forex_candle_contract,
    validate_forex_sequence_contract,
)

OANDA_ENDPOINTS = {
    "practice": "https://api-fxpractice.oanda.com/v3",
    "live": "https://api-fxtrade.oanda.com/v3",
}

SYMBOL_MAP = {
    "EURUSD": "EUR_USD",
    "EURCHF": "EUR_CHF",
}

REVERSE_SYMBOL_MAP = {v: k for k, v in SYMBOL_MAP.items()}


class OandaConfigError(RuntimeError):
    """Raised when required OANDA configuration is absent."""


class OandaAuthError(RuntimeError):
    """Raised when authentication against OANDA fails."""


class OandaForexClient:
    """Isolated research client for OANDA v20 REST market data."""

    def __init__(
        self,
        api_key: str | None = None,
        account_id: str | None = None,
        environment: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("OANDA_API_KEY", "").strip()
        self._account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID", "").strip()
        raw_env = (environment or os.environ.get("OANDA_ENVIRONMENT", "practice")).strip().lower()
        self.environment = raw_env if raw_env in OANDA_ENDPOINTS else "practice"
        self.base_url = OANDA_ENDPOINTS[self.environment]

    @classmethod
    def from_env_or_dotenv(cls) -> OandaForexClient:
        """Loads configuration from os.environ or .env file without printing secrets."""
        key = os.environ.get("OANDA_API_KEY", "").strip()
        acc = os.environ.get("OANDA_ACCOUNT_ID", "").strip()
        env = os.environ.get("OANDA_ENVIRONMENT", "").strip()

        if not (key and acc and env):
            dot_env = Path(".env")
            if dot_env.exists():
                for line in dot_env.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k == "OANDA_API_KEY" and not key:
                            key = v
                        elif k == "OANDA_ACCOUNT_ID" and not acc:
                            acc = v
                        elif k == "OANDA_ENVIRONMENT" and not env:
                            env = v

        return cls(api_key=key, account_id=acc, environment=env or "practice")

    def validate_configuration(self) -> tuple[bool, list[str]]:
        """Checks whether all required parameters are present without exposing values."""
        missing = []
        if not self._api_key:
            missing.append("OANDA_API_KEY")
        if not self._account_id:
            missing.append("OANDA_ACCOUNT_ID")
        if not os.environ.get("OANDA_ENVIRONMENT") and not self.environment:
            missing.append("OANDA_ENVIRONMENT")
        return len(missing) == 0, missing

    def _get_headers(self) -> dict[str, str]:
        if not self._api_key:
            raise OandaConfigError("Missing OANDA_API_KEY: Cannot construct authorization headers.")
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Antigravity-ForexResearch/1.0",
        }

    def test_authenticated_connection(self) -> tuple[bool, str, dict[str, Any]]:
        """Tests authentication against OANDA practice account instruments endpoint."""
        valid, missing = self.validate_configuration()
        if not valid:
            return False, f"Missing configuration: {', '.join(missing)}", {}

        url = f"{self.base_url}/accounts/{self._account_id}/instruments"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")

        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status = resp.status
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                instruments = [i.get("name") for i in data.get("instruments", [])]
                has_eurusd = "EUR_USD" in instruments
                has_eurchf = "EUR_CHF" in instruments
                return True, f"HTTP {status}: Connected. EUR_USD={has_eurusd}, EUR_CHF={has_eurchf}", {
                    "instruments_count": len(instruments),
                    "has_eurusd": has_eurusd,
                    "has_eurchf": has_eurchf,
                }
        except urllib.error.HTTPError as exc:
            err_msg = f"HTTP Error {exc.code}: {exc.reason}"
            return False, err_msg, {}
        except Exception as exc:
            return False, f"Connection failed: {exc}", {}

    def fetch_historical_candles_sample(
        self,
        symbol: str,
        count: int = 50,
        granularity: str = "M5",
        output_dir: Path | str = Path("research/forex_phase11/raw"),
    ) -> tuple[bool, str, list[ForexCandleContractV1], dict[str, Any]]:
        """Retrieves verified historical candles from OANDA v20 REST endpoint.

        Parameters:
        - symbol: 'EURUSD' or 'EURCHF'
        - count: number of completed candles (1..5000)
        - granularity: 'M5' (5-minute bars)
        - output_dir: research directory where immutable raw response and hash are saved
        """
        if symbol in PERMANENT_BLOCKLIST:
            return False, f"Symbol {symbol} is permanently blocked by POL-GOV-2026-001.", [], {}
        if symbol not in APPROVED_FX_SYMBOLS:
            return False, f"Symbol {symbol} is not approved under governance allowlist.", [], {}

        valid, missing = self.validate_configuration()
        if not valid:
            return False, f"Missing configuration: {', '.join(missing)}", [], {}

        oanda_inst = SYMBOL_MAP.get(symbol, symbol)
        url = f"{self.base_url}/instruments/{oanda_inst}/candles?granularity={granularity}&count={count}&price=MBA"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")

        now_utc = datetime.now(UTC)
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                raw_bytes = resp.read()
                raw_text = raw_bytes.decode("utf-8")
                payload = json.loads(raw_text)
        except urllib.error.HTTPError as exc:
            return False, f"HTTP Error {exc.code} during candle retrieval: {exc.reason}", [], {}
        except Exception as exc:
            return False, f"Failed to retrieve candles: {exc}", [], {}

        # Save immutable raw payload
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        ts_tag = now_utc.strftime("%Y%m%d_%H%M%S")
        raw_filename = f"oanda_{symbol}_{granularity}_{ts_tag}.json"
        raw_file = out_path / raw_filename
        raw_file.write_bytes(raw_bytes)

        # Parse and convert to ForexCandleContractV1
        raw_candles = payload.get("candles", [])
        parsed_contracts: list[ForexCandleContractV1] = []
        rejected_observations = 0

        for idx, item in enumerate(raw_candles):
            # Only completed candles are valid for forward modeling
            is_complete = bool(item.get("complete", False))
            if not is_complete:
                rejected_observations += 1
                continue

            time_str = item.get("time", "")
            mid = item.get("mid", {})
            bid = item.get("bid", {})
            ask = item.get("ask", {})

            try:
                open_p = float(mid["o"])
                high_p = float(mid["h"])
                low_p = float(mid["l"])
                close_p = float(mid["c"])
                vol = float(item.get("volume", 0.0))

                bid_c = float(bid.get("c", 0.0)) if bid else None
                ask_c = float(ask.get("c", 0.0)) if ask else None
                spread_val = (ask_c - bid_c) if (ask_c is not None and bid_c is not None) else None

                contract = ForexCandleContractV1(
                    schema_version="FX-DATA-CONTRACT-v1.0.0",
                    symbol=symbol,
                    provider="OANDA_V20_REST",
                    native_interval="5m",
                    price_basis=PriceBasis.MID,
                    timestamp_utc=time_str,
                    retrieved_at_utc=now_utc.isoformat(),
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=vol,
                    is_closed=True,
                    bid_close=bid_c,
                    ask_close=ask_c,
                    spread=spread_val,
                    source_record_id=f"oanda_{oanda_inst}_{time_str}",
                )
                valid, msg = validate_forex_candle_contract(contract)
                if valid:
                    parsed_contracts.append(contract)
                else:
                    rejected_observations += 1
            except (KeyError, ValueError, TypeError):
                rejected_observations += 1

        seq_valid, seq_msg, detected_gaps = validate_forex_sequence_contract(parsed_contracts, expected_interval="5m")

        metadata = {
            "retrieval_timestamp_utc": now_utc.isoformat(),
            "symbol": symbol,
            "oanda_instrument": oanda_inst,
            "granularity": granularity,
            "requested_count": count,
            "raw_candles_received": len(raw_candles),
            "validated_contracts_count": len(parsed_contracts),
            "rejected_observations_count": rejected_observations,
            "detected_gaps_count": len(detected_gaps),
            "sequence_valid": seq_valid,
            "sequence_validation_message": seq_msg,
            "raw_file_path": str(raw_file),
            "raw_sha256": raw_hash,
        }

        meta_file = out_path / f"oanda_{symbol}_{granularity}_{ts_tag}_metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return True, f"Successfully parsed {len(parsed_contracts)} contracts (hash: {raw_hash[:16]}...)", parsed_contracts, metadata
