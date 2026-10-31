"""Tests for Isolated OANDA v20 Research Client.

Verifies:
1. Rejection of unconfigured client without exposing secrets.
2. Safe parsing of valid M5 OANDA payload with discrete bid/ask.
3. Filtering of incomplete candles (complete=false).
4. Enforcement of governance blocklist (EURAUD, EURCAD).
5. Sequence contract validation and timestamp ordering.
6. Test isolation from production ledgers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import pytest

from data.contracts.forex_contract import PriceBasis
from research.forex_phase11.oanda_client import (
    OandaConfigError,
    OandaForexClient,
)


def test_oanda_client_unconfigured_rejection(monkeypatch):
    """Test 1: Unconfigured client safely detects missing credentials without exposing secrets."""
    monkeypatch.delenv("OANDA_API_KEY", raising=False)
    monkeypatch.delenv("OANDA_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("OANDA_ENVIRONMENT", raising=False)

    client = OandaForexClient(api_key="", account_id="", environment="")
    is_valid, missing = client.validate_configuration()

    assert is_valid is False
    assert "OANDA_API_KEY" in missing
    assert "OANDA_ACCOUNT_ID" in missing

    with pytest.raises(OandaConfigError, match="Missing OANDA_API_KEY"):
        client._get_headers()


def test_oanda_client_blocklist_enforcement():
    """Test 2: Blocklisted symbols are rejected before any API request is made."""
    client = OandaForexClient(api_key="mock_key", account_id="mock_acc", environment="practice")

    for blocked in ["EURAUD", "EURCAD"]:
        ok, msg, contracts, meta = client.fetch_historical_candles_sample(symbol=blocked)
        assert ok is False
        assert "permanently blocked" in msg
        assert len(contracts) == 0


def test_oanda_payload_parsing_and_contract_validation(tmp_path, monkeypatch):
    """Test 3: Validates parsing of realistic OANDA M5 payload with discrete bid/ask."""
    base_t = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)

    # Mock OANDA v20 JSON response with 5 candles (4 complete, 1 incomplete)
    mock_payload = {
        "instrument": "EUR_USD",
        "granularity": "M5",
        "candles": [
            {
                "complete": True,
                "volume": 120,
                "time": (base_t + timedelta(minutes=5 * i)).isoformat(),
                "mid": {
                    "o": f"{1.1200 + 0.0001 * i:.5f}",
                    "h": f"{1.1205 + 0.0001 * i:.5f}",
                    "l": f"{1.1195 + 0.0001 * i:.5f}",
                    "c": f"{1.1202 + 0.0001 * i:.5f}",
                },
                "bid": {
                    "o": f"{1.11992 + 0.0001 * i:.5f}",
                    "h": f"{1.12042 + 0.0001 * i:.5f}",
                    "l": f"{1.11942 + 0.0001 * i:.5f}",
                    "c": f"{1.12012 + 0.0001 * i:.5f}",
                },
                "ask": {
                    "o": f"{1.12008 + 0.0001 * i:.5f}",
                    "h": f"{1.12058 + 0.0001 * i:.5f}",
                    "l": f"{1.11958 + 0.0001 * i:.5f}",
                    "c": f"{1.12028 + 0.0001 * i:.5f}",
                },
            }
            for i in range(4)
        ] + [
            # Incomplete in-flight candle at index 4 (MUST be rejected)
            {
                "complete": False,
                "volume": 25,
                "time": (base_t + timedelta(minutes=20)).isoformat(),
                "mid": {"o": "1.12040", "h": "1.12060", "l": "1.12030", "c": "1.12050"},
                "bid": {"o": "1.12032", "h": "1.12052", "l": "1.12022", "c": "1.12042"},
                "ask": {"o": "1.12048", "h": "1.12068", "l": "1.12038", "c": "1.12058"},
            }
        ]
    }

    raw_bytes = json.dumps(mock_payload).encode("utf-8")

    class MockResponse:
        def __init__(self, data):
            self.data = data
            self.status = 200

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=15.0: MockResponse(raw_bytes))

    client = OandaForexClient(api_key="mock_token", account_id="mock_acc", environment="practice")
    res_dir = tmp_path / "raw_oanda"

    ok, msg, contracts, meta = client.fetch_historical_candles_sample(
        symbol="EURUSD", count=5, granularity="M5", output_dir=res_dir
    )

    assert ok is True
    assert "Successfully parsed 4 contracts" in msg
    # The 5th incomplete candle was rejected
    assert len(contracts) == 4
    assert meta["rejected_observations_count"] == 1
    assert meta["validated_contracts_count"] == 4
    assert meta["sequence_valid"] is True

    # Validate contracts
    c0 = contracts[0]
    assert c0.symbol == "EURUSD"
    assert c0.provider == "OANDA_V20_REST"
    assert c0.price_basis == PriceBasis.MID
    assert c0.is_closed is True
    assert c0.ask_close > c0.bid_close
    assert round(c0.spread, 6) == 0.000160

    # Verify raw file and metadata was written to tmp_path
    assert Path(meta["raw_file_path"]).exists()
    assert len(meta["raw_sha256"]) == 64

    # Verify production paths were NOT touched
    assert not Path("data/ledger/EURUSD.json").exists()
