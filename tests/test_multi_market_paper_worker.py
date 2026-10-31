"""Unit tests for Multi-Market Provider-Neutral Paper Trading Worker Architecture."""

from datetime import UTC, datetime
from pathlib import Path
import pytest

from core.events import Candle, RawMarketEvent
from paper_trading.worker import PaperTradingWorker, WorkerState


@pytest.fixture
def temp_worker_env(tmp_path):
    store_dir = tmp_path / "paper_worker_test"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def test_multi_market_provider_routing(temp_worker_env):
    """Verify that paper worker correctly routes instruments to their respective exchange providers."""
    worker = PaperTradingWorker(
        store_dir=temp_worker_env,
        instruments=["NSE:TCS", "BSE:RELIANCE", "QUOTEX:BTCUSD_QX", "EURUSD"],
    )

    p_nse, sym_nse, mkt_nse, proxy_nse = worker._get_provider_for_instrument("NSE:TCS")
    assert mkt_nse == "NSE"
    assert sym_nse == "TCS"
    assert proxy_nse is False

    p_bse, sym_bse, mkt_bse, proxy_bse = worker._get_provider_for_instrument("BSE:RELIANCE")
    assert mkt_bse == "BSE"
    assert sym_bse == "RELIANCE"
    assert proxy_bse is False

    p_qx, sym_qx, mkt_qx, proxy_qx = worker._get_provider_for_instrument("QUOTEX:BTCUSD_QX")
    assert mkt_qx == "QUOTEX"
    assert sym_qx == "BTCUSD_QX"
    assert proxy_qx is True

    p_fx, sym_fx, mkt_fx, proxy_fx = worker._get_provider_for_instrument("EURUSD")
    assert mkt_fx == "FOREX"
    assert sym_fx == "EURUSD"
    assert proxy_fx is True


def test_quotex_otc_rejection(temp_worker_env):
    """Verify that synthetic Quotex OTC instruments with no authorized public feed are rejected."""
    worker = PaperTradingWorker(
        store_dir=temp_worker_env,
        instruments=["QUOTEX:MSFT_OTC"],
    )
    now_dt = datetime.now(UTC)
    res = worker._process_instrument_prediction("QUOTEX:MSFT_OTC", now_dt)
    assert res["prediction_recorded"] is False
    assert "QUOTEX_DATA_UNAVAILABLE" in res["reason"] or "disabled by governance" in res["reason"]


def test_unsupported_forex_rejection(temp_worker_env):
    """Verify that blocked/unauthenticated Forex instruments are rejected by governance."""
    worker = PaperTradingWorker(
        store_dir=temp_worker_env,
        instruments=["FX:EURUSD"],
    )
    now_dt = datetime.now(UTC)
    res = worker._process_instrument_prediction("FX:EURUSD", now_dt)
    assert res["prediction_recorded"] is False
    assert "disabled by governance" in res["reason"] or "prohibited" in res["reason"]


def test_prospective_storage_isolation(temp_worker_env, monkeypatch):
    """Verify that prospective paper predictions are written strictly to paper storage."""
    worker = PaperTradingWorker(
        store_dir=temp_worker_env,
        instruments=["NSE:TCS"],
    )
    assert worker.initialize_model() is True

    # 10:30 AM IST (05:00 UTC) during active trading hours
    now_dt = datetime(2026, 10, 20, 5, 0, 0, tzinfo=UTC)

    # Mock live quote
    mock_quote = RawMarketEvent(
        provider="TEST",
        symbol="TCS",
        channel="quote",
        received_at=now_dt,
        raw_payload={
            "type": "quote",
            "exchange": "NSE",
            "symbol": "TCS",
            "price": 3500.0,
            "bid": 3499.5,
            "ask": 3500.5,
            "timestamp": now_dt.timestamp(),
            "source": "TEST_NSE",
        },
    )

    # Mock historical candles
    import pandas as pd
    mock_candles = [
        {
            "timestamp": (now_dt - pd.Timedelta(minutes=5 * (40 - i))).isoformat(),
            "open": 3500.0 + i,
            "high": 3510.0 + i,
            "low": 3490.0 + i,
            "close": 3500.0 + i,
            "volume": 1000.0,
            "is_closed": True,
        }
        for i in range(40)
    ]

    monkeypatch.setattr(worker.nse_provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_quote)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: mock_candles)

    # Run cycle
    res = worker._process_instrument_prediction("NSE:TCS", now_dt)
    assert res["prediction_recorded"] is True, f"Prediction was not recorded: {res.get('reason')}"

    # Verify written to temp store
    paper_preds_file = temp_worker_env / "paper_predictions.jsonl"
    assert paper_preds_file.exists()
    assert paper_preds_file.read_text().strip() != ""
