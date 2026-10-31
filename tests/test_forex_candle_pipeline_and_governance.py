"""Comprehensive Integration and Regression Tests for Forex Candle Pipeline & Governance.

Verifies Phase 4 requirements:
1. Disabled instruments cannot generate predictions.
2. Stale candles cannot generate predictions even when live quotes are fresh.
3. Missing candle data cannot fall back silently to an old cache.
4. A diagnostic or cache-refresh command cannot inadvertently create a prospective prediction.
5. Restarting the worker preserves the same restrictions.
6. An invalid instrument cannot affect the status or eligibility of a valid instrument.
7. A provider outage cannot cause stale data to be used as a fallback.
8. ECB reference rates are strictly daily ('1d'), never fabricated into 5m candles.
9. Aggregation rejects downsampling of daily series into intraday bars.
10. 5m candles fail closed when a genuine intraday provider is unavailable.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
import pytest

from core.events import Candle, RawMarketEvent
from data.ingestion.forex import (
    ForexDataProvider,
    aggregate_forex_candles,
    validate_candle_sequence,
    validate_forex_candle,
)
from paper_trading.prediction_store import PaperTradingPredictionStore
from paper_trading.worker import (
    ACTIVE_GOVERNANCE_ALLOWLIST,
    EXPLICIT_BLOCKLIST,
    PaperTradingWorker,
    WorkerState,
    get_applicable_freshness_threshold,
)


@pytest.fixture
def temp_store_dir(tmp_path):
    store = tmp_path / "paper_test_store"
    store.mkdir(parents=True, exist_ok=True)
    return store


def test_disabled_instrument_blocked_by_governance(temp_store_dir, monkeypatch):
    """Test 1: Proves that disabled instruments (EURAUD, EURCAD) are blocked inside the worker."""
    worker = PaperTradingWorker(
        store_dir=temp_store_dir,
        instruments=["EURAUD", "EURCAD"],
    )
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0

    for blocked_sym in ["EURAUD", "EURCAD"]:
        status = res["instruments"][blocked_sym]
        assert status["prediction_recorded"] is False
        assert status["is_eligible"] is False
        assert "disabled by governance policy" in status["reason"]
        assert blocked_sym in worker.blocklist

    assert len(worker.store.get_pending_predictions()) == 0


def test_stale_candles_cannot_generate_prediction_with_fresh_quotes(temp_store_dir, monkeypatch):
    """Test 2: Stale candles cannot generate predictions even when live quotes are fresh."""
    worker = PaperTradingWorker(
        store_dir=temp_store_dir,
        instruments=["EURUSD"],
    )
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)

    # Mock fresh quote (0 seconds old)
    now_dt = datetime.now(UTC)
    mock_quote = RawMarketEvent(
        provider="forex_provider",
        symbol="EURUSD",
        channel="spread",
        received_at=now_dt,
        raw_payload={
            "type": "quote",
            "symbol": "EURUSD",
            "price": 1.1230,
            "bid": 1.1229,
            "ask": 1.1231,
            "spread": 0.0002,
            "timestamp": now_dt.timestamp(),
            "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_quote)

    # Mock stale candles (exceeds both weekday 36h and weekend 76h limit)
    stale_dt = now_dt - timedelta(hours=80)
    mock_candles = [
        {
            "timestamp": (stale_dt - timedelta(days=i)).isoformat(),
            "open": 1.1200,
            "high": 1.1250,
            "low": 1.1180,
            "close": 1.1220,
            "volume": 1000.0,
        }
        for i in range(35, -1, -1)
    ]
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: mock_candles)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    inst_res = res["instruments"]["EURUSD"]
    assert inst_res["quote_fresh"] is True
    assert inst_res["candle_fresh"] is False
    assert inst_res["prediction_recorded"] is False
    assert "STALE" in inst_res["reason"]
    assert worker.state == WorkerState.DEGRADED
    assert len(worker.store.get_pending_predictions()) == 0


def test_missing_candle_data_does_not_silently_fallback_to_stale_cache(temp_store_dir, monkeypatch):
    """Test 3: Missing candle data fails closed and does NOT fall back silently to old cache."""
    worker = PaperTradingWorker(
        store_dir=temp_store_dir,
        instruments=["EURUSD"],
    )
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)

    now_dt = datetime.now(UTC)
    mock_quote = RawMarketEvent(
        provider="forex_provider",
        symbol="EURUSD",
        channel="spread",
        received_at=now_dt,
        raw_payload={
            "type": "quote",
            "symbol": "EURUSD",
            "price": 1.1230,
            "bid": 1.1229,
            "ask": 1.1231,
            "spread": 0.0002,
            "timestamp": now_dt.timestamp(),
            "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_quote)

    # Return empty candles (missing data / provider outage)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: [])

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    inst_res = res["instruments"]["EURUSD"]
    assert inst_res["prediction_recorded"] is False
    assert "Insufficient historical candle history" in inst_res["reason"]
    assert len(worker.store.get_pending_predictions()) == 0


def test_diagnostic_or_cache_refresh_cannot_create_prospective_prediction(temp_store_dir):
    """Test 4: Diagnostic or cache refresh commands cannot inadvertently create prospective predictions."""
    store = PaperTradingPredictionStore(store_dir=temp_store_dir)
    assert len(store.get_pending_predictions()) == 0

    provider = ForexDataProvider()
    # Execute cache refresh diagnostics
    success, count, msg = provider.refresh_fx_candle_cache("EURUSD", timeframe="1d")

    # Store must remain strictly 0 pending predictions
    assert len(store.get_pending_predictions()) == 0
    assert not (temp_store_dir / "paper_predictions.jsonl").exists() or len(store.get_pending_predictions()) == 0


def test_restarting_worker_preserves_allowlist_and_restrictions(temp_store_dir):
    """Test 5: Restarting the worker preserves identical allowlist, blocklist, and governance gates."""
    w1 = PaperTradingWorker(store_dir=temp_store_dir)
    assert w1.allowlist == ACTIVE_GOVERNANCE_ALLOWLIST
    assert w1.blocklist == EXPLICIT_BLOCKLIST
    assert "EURAUD" in w1.blocklist
    assert "EURCAD" in w1.blocklist
    assert "EURUSD" in w1.allowlist
    assert "EURCHF" in w1.allowlist

    # Simulate restart
    w2 = PaperTradingWorker(store_dir=temp_store_dir)
    assert w2.allowlist == w1.allowlist
    assert w2.blocklist == w1.blocklist
    assert "EURUSD" in w2.allowlist
    assert "EURCHF" in w2.allowlist
    assert "EURAUD" in w2.blocklist
    assert "EURCAD" in w2.blocklist


def test_invalid_instrument_does_not_affect_valid_instrument(temp_store_dir, monkeypatch):
    """Test 6: An invalid instrument (EURAUD) cannot affect the eligibility or status of a valid instrument (EURUSD)."""
    worker = PaperTradingWorker(
        store_dir=temp_store_dir,
        instruments=["EURUSD", "EURAUD"],
    )
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)

    now_dt = datetime.now(UTC)
    mock_quote = RawMarketEvent(
        provider="forex_provider",
        symbol="EURUSD",
        channel="spread",
        received_at=now_dt,
        raw_payload={
            "type": "quote",
            "symbol": "EURUSD",
            "price": 1.1230,
            "bid": 1.1229,
            "ask": 1.1231,
            "spread": 0.0002,
            "timestamp": now_dt.timestamp(),
            "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_quote if sym == "EURUSD" else None)

    # Valid candles for EURUSD
    mock_candles_eurusd = [
        {
            "timestamp": (now_dt - timedelta(days=i, hours=12)).isoformat(),
            "open": 1.1200,
            "high": 1.1250,
            "low": 1.1180,
            "close": 1.1220,
            "volume": 1000.0,
        }
        for i in range(35, -1, -1)
    ]
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: mock_candles_eurusd if sym == "EURUSD" else [])

    res = worker.poll_once()
    # EURUSD succeeded in evaluating (fresh, valid)
    assert res["instruments"]["EURUSD"]["quote_fresh"] is True
    assert res["instruments"]["EURUSD"]["candle_fresh"] is True

    # EURAUD was blocked independently by governance
    assert res["instruments"]["EURAUD"]["prediction_recorded"] is False
    assert res["instruments"]["EURAUD"]["is_eligible"] is False
    assert "disabled by governance policy" in res["instruments"]["EURAUD"]["reason"]


def test_provider_outage_fails_closed_without_stale_fallback(monkeypatch):
    """Test 7: Provider outage fails closed without using stale data as fallback."""
    provider = ForexDataProvider()
    # Simulate network outage: all HTTP calls fail
    monkeypatch.setattr(provider, "_fetch_daily_ecb_rates", lambda sym, limit=360, timeout=4.0: [])

    success, count, msg = provider.refresh_fx_candle_cache("EURUSD", timeframe="1d")
    assert success is False
    assert count == 0
    assert "Provider outage" in msg


def test_ecb_reference_rates_strictly_daily_timeframe():
    """Test 8: ECB reference rates are strictly daily ('1d') observations."""
    dt = datetime.now(UTC)
    c_daily = Candle(
        symbol="EURUSD",
        timeframe="1d",
        timestamp=dt,
        open=1.1225,
        high=1.1225,
        low=1.1225,
        close=1.1225,
        volume=0.0,
        source_provider="ECB_FRANKFURTER",
        source_interval="1d",
    )
    valid, msg = validate_forex_candle(c_daily)
    assert valid is True
    assert msg == "VALID"
    assert c_daily.timeframe == "1d"
    assert c_daily.source_interval == "1d"
    assert c_daily.source_provider == "ECB_FRANKFURTER"


def test_aggregation_refuses_daily_downsampling():
    """Test 9: Aggregator rejects downsampling daily series into intraday bars."""
    dt = datetime.now(UTC)
    candles = [
        Candle(
            symbol="EURUSD",
            timeframe="1d",
            timestamp=dt - timedelta(days=i),
            open=1.12,
            high=1.13,
            low=1.11,
            close=1.125,
            volume=0.0,
        )
        for i in range(10, -1, -1)
    ]
    with pytest.raises(ValueError, match="Cannot downsample daily reference observations"):
        aggregate_forex_candles(candles, target_tf="5m", symbol="EURUSD")

    with pytest.raises(ValueError, match="Cannot downsample daily reference observations"):
        aggregate_forex_candles(candles, target_tf="1h", symbol="EURUSD")


def test_intraday_5m_pipeline_fails_closed_when_provider_unavailable():
    """Test 10: 5m candle fetching fails closed when genuine intraday data is unavailable."""
    provider = ForexDataProvider()
    res = provider.fetch_historical_candles_sync("EURUSD", timeframe="5m")
    assert res == []  # Fails closed, zero fake sine-wave candles


def test_mathematical_aggregation_factors_and_invariants():
    """Test 11: Mathematical aggregation strictly enforces 3, 12, 48, 288 bar factors and OHLC invariants."""
    base_time = datetime(2026, 10, 19, 0, 0, tzinfo=UTC)
    # Generate 300 5-minute bars (25 hours)
    bars_5m = [
        Candle(
            symbol="EURUSD",
            timeframe="5m",
            timestamp=base_time + timedelta(minutes=5 * i),
            open=1.1200 + (i % 10) * 0.0001,
            high=1.1210 + (i % 10) * 0.0001,
            low=1.1190 + (i % 10) * 0.0001,
            close=1.1205 + (i % 10) * 0.0001,
            volume=100.0,
            is_closed=True,
        )
        for i in range(300)
    ]

    # 1. 5m -> 15m (factor 3): 300 / 3 = 100 complete bars
    bars_15m = aggregate_forex_candles(bars_5m, target_tf="15m", symbol="EURUSD", require_complete=True)
    assert len(bars_15m) == 100
    assert bars_15m[0].open == bars_5m[0].open
    assert bars_15m[0].close == bars_5m[2].close
    assert bars_15m[0].volume == 300.0
    assert bars_15m[0].timeframe == "15m"

    # 2. 5m -> 1h (factor 12): 300 / 12 = 25 complete bars
    bars_1h = aggregate_forex_candles(bars_5m, target_tf="1h", symbol="EURUSD", require_complete=True)
    assert len(bars_1h) == 25
    assert bars_1h[0].open == bars_5m[0].open
    assert bars_1h[0].close == bars_5m[11].close
    assert bars_1h[0].volume == 1200.0
    assert bars_1h[0].timeframe == "1h"

    # 3. 5m -> 4h (factor 48): 300 // 48 = 6 complete bars (trailing 12 bars dropped)
    bars_4h = aggregate_forex_candles(bars_5m, target_tf="4h", symbol="EURUSD", require_complete=True)
    assert len(bars_4h) == 6
    assert bars_4h[0].open == bars_5m[0].open
    assert bars_4h[0].close == bars_5m[47].close
    assert bars_4h[0].volume == 4800.0

    # 4. 5m -> 1d (factor 288): 300 // 288 = 1 complete bar (trailing 12 bars dropped)
    bars_1d = aggregate_forex_candles(bars_5m, target_tf="1d", symbol="EURUSD", require_complete=True)
    assert len(bars_1d) == 1
    assert bars_1d[0].open == bars_5m[0].open
    assert bars_1d[0].close == bars_5m[287].close
    assert bars_1d[0].volume == 28800.0


def test_intraday_timestamp_in_daily_sequence_rejected():
    """Test 12: Sequence interval validator strictly detects and rejects intraday bars in daily series."""
    base_dt = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
    daily_candles = [
        Candle(
            symbol="EURUSD",
            timeframe="1d",
            timestamp=base_dt + timedelta(days=i),
            open=1.1200,
            high=1.1250,
            low=1.1180,
            close=1.1220,
            volume=0.0,
            source_provider="ECB_FRANKFURTER",
            source_interval="1d",
        )
        for i in range(30)
    ]

    # Valid daily sequence passes
    valid, msg = validate_candle_sequence(daily_candles, expected_timeframe="1d")
    assert valid is True
    assert msg == "VALID"

    # Insert an intraday bar (7.86 hours later, simulating TradingView snapshot)
    contaminated = list(daily_candles)
    intraday_ts = daily_candles[-1].timestamp + timedelta(hours=7, minutes=51, seconds=32)
    contaminated.append(
        Candle(
            symbol="EURUSD",
            timeframe="1d",
            timestamp=intraday_ts,
            open=1.1230,
            high=1.1235,
            low=1.1228,
            close=1.1231,
            volume=150.0,
            source_provider="TRADINGVIEW",
            source_interval="live",
        )
    )

    valid_c, msg_c = validate_candle_sequence(contaminated, expected_timeframe="1d")
    assert valid_c is False
    assert "Sequence interval violation" in msg_c
    assert "Intraday bar detected in daily sequence" in msg_c


def test_governance_tampering_resistance(temp_store_dir):
    """Test 13: Worker cannot be coerced into enabling blocked or unapproved instruments."""
    # Attempt to bypass governance by clearing blocklist and adding EURAUD to allowlist
    worker = PaperTradingWorker(
        store_dir=temp_store_dir,
        allowlist=["EURAUD", "EURCAD", "UNAPPROVED_PAIR", "EURUSD"],
        blocklist=[],
    )

    # Blocklist must still enforce permanent blocks
    assert "EURAUD" in worker.blocklist
    assert "EURCAD" in worker.blocklist

    # Allowlist must have filtered out blocked and unapproved pairs
    assert "EURAUD" not in worker.allowlist
    assert "EURCAD" not in worker.allowlist
    assert "UNAPPROVED_PAIR" not in worker.allowlist
    assert worker.allowlist == ["EURUSD"]

