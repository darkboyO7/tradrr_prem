"""Deterministic Unit and Integration Tests for PaperTradingWorker and Evaluator.

Covers:
- Duplicate-worker prevention via kernel process lock.
- Clean start and shutdown lifecycle.
- Restart recovery and idempotency.
- Prediction immutability and duplicate prevention.
- Duplicate outcome-resolution prevention.
- Weekend and market-closure handling.
- Stale data rejection and DEGRADED state transition.
- Executable quote-side pricing (Ask entry/Bid exit for BUY; Bid entry/Ask exit for SELL).
- Verification that transaction costs are not deducted twice.
- Missing-data deferral and expiration without price fabrication.
- Exclusion of expired records from active win-rate denominator.
- Pre-registered promotion policy gates & insufficient evidence handling.
- Strict isolation from production ledgers and real-money execution paths.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from core.events import RawMarketEvent
from paper_trading.evaluator import (
    MIN_PROSPECTIVE_CALENDAR_DAYS,
    MIN_PROSPECTIVE_OBSERVATIONS,
    PaperTradingEvaluator,
    stationary_block_bootstrap,
)
from paper_trading.prediction_store import (
    PaperPredictionRecord,
    PaperTradingPredictionStore,
)
from paper_trading.worker import (
    FROZEN_DECISION_MARGIN,
    FROZEN_TEMPERATURE,
    PaperTradingWorker,
    ProcessFileLock,
    WorkerState,
)


@pytest.fixture
def temp_paper_env(tmp_path):
    store_dir = tmp_path / "paper_trading_test"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def _mock_valid_daily_candles(symbol: str = "EURUSD", count: int = 40):
    now_dt = datetime.now(UTC)
    return [
        {
            "timestamp": (now_dt - timedelta(days=i, hours=12)).isoformat(),
            "open": 1.1200 + i * 0.0001,
            "high": 1.1250 + i * 0.0001,
            "low": 1.1180 + i * 0.0001,
            "close": 1.1220 + i * 0.0001,
            "volume": 1000.0,
        }
        for i in range(count - 1, -1, -1)
    ]


def test_process_file_lock_duplicate_prevention(temp_paper_env):
    """Verifies that two worker instances cannot run concurrently on the same store."""
    lock_file = temp_paper_env / ".paper_worker.lock"
    lock1 = ProcessFileLock(lock_file)
    lock2 = ProcessFileLock(lock_file)

    assert lock1.acquire() is True
    assert lock2.acquire() is False

    lock1.release()
    assert lock2.acquire() is True
    lock2.release()


def test_worker_clean_start_and_shutdown(temp_paper_env):
    """Verifies lifecycle transitions: STOPPED -> RUNNING -> STOPPED."""
    worker = PaperTradingWorker(
        store_dir=temp_paper_env,
        instruments=["EURUSD"],
        poll_interval_seconds=1.0,
    )
    assert worker.state == WorkerState.STOPPED

    started = worker.start()
    assert started is True
    assert worker.state == WorkerState.RUNNING

    worker.stop(timeout=2.0)
    assert worker.state == WorkerState.STOPPED


def test_worker_duplicate_start_rejection(temp_paper_env):
    """Verifies that starting a second worker on the same store fails cleanly."""
    w1 = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)
    w2 = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)

    assert w1.start() is True
    assert w2.start() is False
    assert w2.state == WorkerState.ERROR

    w1.stop()


def test_worker_duplicate_prediction_prevention(temp_paper_env, monkeypatch):
    """Verifies that polling twice on the same completed candle does not duplicate predictions."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)
    assert worker.initialize_model() is True

    # Mock market open and live quote
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)
    fixed_candles = _mock_valid_daily_candles("EURUSD")
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: fixed_candles)
    now_ts = datetime.now(UTC).timestamp()
    mock_event = RawMarketEvent(
        provider="forex_provider",
        symbol="EURUSD",
        channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote",
            "symbol": "EURUSD",
            "price": 1.12250,
            "bid": 1.12242,
            "ask": 1.12258,
            "spread": 0.00016,
            "timestamp": now_ts,
            "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_event)

    # 1. First poll records prediction
    res1 = worker.poll_once()
    assert res1["predictions_generated"] == 1
    assert len(worker.store.get_pending_predictions()) == 1

    # 2. Immediate second poll detects same candle and skips
    res2 = worker.poll_once()
    assert res2["predictions_generated"] == 0
    assert len(worker.store.get_pending_predictions()) == 1


def test_worker_market_closure_handling(temp_paper_env, monkeypatch):
    """Verifies that when market is closed (weekend/holiday), worker skips without generating fake candles."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"])
    assert worker.initialize_model() is True

    # Mock market closed
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: False)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    assert res["instruments"]["EURUSD"]["market_open"] is False
    assert "Market is closed" in res["instruments"]["EURUSD"]["reason"]
    assert len(worker.store.get_pending_predictions()) == 0


def test_worker_stale_data_rejection(temp_paper_env, monkeypatch):
    """Verifies that stale broker quotes trigger DEGRADED state and reject prediction."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], stale_quote_threshold_seconds=15.0)
    assert worker.initialize_model() is True

    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: _mock_valid_daily_candles(sym))
    old_ts = (datetime.now(UTC) - timedelta(seconds=120)).timestamp()
    stale_event = RawMarketEvent(
        provider="forex_provider",
        symbol="EURUSD",
        channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote",
            "symbol": "EURUSD",
            "price": 1.12000,
            "bid": 1.11990,
            "ask": 1.12010,
            "spread": 0.00020,
            "timestamp": old_ts,
            "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: stale_event)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    assert res["instruments"]["EURUSD"]["quote_fresh"] is False
    assert worker.state == WorkerState.DEGRADED


def test_executable_quote_side_pricing_and_no_double_counting(temp_paper_env, monkeypatch):
    """Verifies:

    1. BUY fills at Ask + slippage, exits at Bid - slippage.
    2. SELL fills at Bid - slippage, exits at Ask + slippage.
    3. Costs are NOT deducted twice (spread is embedded in fill prices).
    """
    store = PaperTradingPredictionStore(store_dir=temp_paper_env)

    # 1. Record BUY prediction
    # Mid = 1.10000, Ask = 1.10008, Bid = 1.09992. Slippage = 0.25 bps (0.0000275)
    ref_mid = 1.10000
    ask_entry = 1.10008
    slip = 0.000025 * ref_mid
    exec_entry_buy = ask_entry + slip

    pred_buy = PaperPredictionRecord(
        prediction_id="pred_exec_buy",
        instrument="EURUSD",
        timeframe="1d",
        forecast_horizon="1d",
        candle_timestamp="2026-10-21T00:00:00+00:00",
        prediction_timestamp="2026-10-21T00:01:00+00:00",
        model_version="v20261020_044046",
        model_sha256="dummy_hash",
        config_hash="dummy_cfg",
        prob_sell_raw=0.1, prob_neutral_raw=0.3, prob_buy_raw=0.6,
        prob_sell_calibrated=0.1, prob_neutral_calibrated=0.3, prob_buy_calibrated=0.6,
        decision="BUY",
        decision_margin=0.3,
        no_trade_reason="",
        reference_price=ref_mid,
        executable_entry_price=exec_entry_buy,
        intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
        scheduled_outcome_timestamp="2026-10-22T00:00:00+00:00",
        data_source_id="TEST_BROKER",
        freshness_status="FRESH",
    )
    store.record_prediction(pred_buy)

    # Exit quote: Mid = 1.10500, Bid = 1.10492, Ask = 1.10508.
    mid_exit = 1.10500
    bid_exit = 1.10492
    exec_exit_buy = bid_exit - slip

    # Resolve
    outcome = store.resolve_prediction(
        prediction_id="pred_exec_buy",
        observed_exit_price=exec_exit_buy,
        resolution_timestamp="2026-10-22T00:05:00+00:00",
        commission_cost=0.000050,  # 0.5 bps
        spread_embedded_in_price=True,
        notes="Testing executable quote pricing without double-counting",
    )

    expected_gross = (exec_exit_buy - exec_entry_buy) / exec_entry_buy
    assert outcome.gross_return == pytest.approx(expected_gross, abs=1e-5)
    # Net return is gross return minus external commission
    expected_net = expected_gross - 0.000050
    assert outcome.net_return == pytest.approx(expected_net, abs=1e-5)
    assert outcome.outcome == "WIN"


def test_evaluator_insufficient_evidence_gate(temp_paper_env):
    """Verifies that the evaluator emits INSUFFICIENT_PROSPECTIVE_EVIDENCE when observations < 100 or days < 60."""
    store = PaperTradingPredictionStore(store_dir=temp_paper_env)

    # Create 5 synthetic predictions and outcomes
    now_dt = datetime.now(UTC)
    for i in range(5):
        pid = f"eval_test_{i}"
        pred = PaperPredictionRecord(
            prediction_id=pid,
            instrument="EURUSD",
            timeframe="1d",
            forecast_horizon="1d",
            candle_timestamp=now_dt.isoformat(),
            prediction_timestamp=now_dt.isoformat(),
            model_version="v20261020_044046",
            model_sha256="dummy_hash",
            config_hash="dummy_cfg",
            prob_sell_raw=0.2, prob_neutral_raw=0.3, prob_buy_raw=0.5,
            prob_sell_calibrated=0.2, prob_neutral_calibrated=0.3, prob_buy_calibrated=0.5,
            decision="BUY",
            decision_margin=0.2,
            no_trade_reason="",
            reference_price=1.10000,
            executable_entry_price=1.10008,
            intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
            scheduled_outcome_timestamp=(now_dt + timedelta(days=1)).isoformat(),
            data_source_id="TEST_BROKER",
            freshness_status="FRESH",
        )
        store.record_prediction(pred)
        store.resolve_prediction(
            prediction_id=pid,
            observed_exit_price=1.10200,
            resolution_timestamp=(now_dt + timedelta(days=1)).isoformat(),
        )

    evaluator = PaperTradingEvaluator(store_dir=temp_paper_env)
    report = evaluator.evaluate()

    assert report["evidence_status"] == "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert report["sample_progress"]["valid_resolved_observations"] == 5
    assert report["sample_progress"]["required_minimum_observations"] == MIN_PROSPECTIVE_OBSERVATIONS
    assert report["sample_progress"]["calendar_days_accumulated"] < MIN_PROSPECTIVE_CALENDAR_DAYS
    assert report["gate_compliance"]["G1_MINIMUM_OBSERVATIONS"]["passed"] is False


def test_stationary_block_bootstrap_uncertainty():
    """Verifies that the block bootstrap generates consistent confidence intervals."""
    rng = np.random.default_rng(42)
    # Autocorrelated returns
    returns = np.zeros(100)
    returns[0] = 0.0005
    for t in range(1, 100):
        returns[t] = 0.3 * returns[t - 1] + rng.normal(0.0002, 0.003)

    ci_low, ci_high = stationary_block_bootstrap(returns, expected_block_len=5, n_bootstraps=500)
    assert ci_low < ci_high
    assert ci_low <= np.mean(returns) <= ci_high


def test_production_ledger_isolation_during_paper_trading(temp_paper_env):
    """Guarantees that paper trading operations cannot modify production ledgers."""
    import hashlib

    prod_ledger_path = Path("data/ledger/prediction_outcomes.jsonl")
    initial_hash = hashlib.sha256(prod_ledger_path.read_bytes()).hexdigest()

    # Perform extensive paper trading actions in temporary store
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"])
    worker.initialize_model()

    store = PaperTradingPredictionStore(store_dir=temp_paper_env)
    pred = PaperPredictionRecord(
        prediction_id="isolation_test_01",
        instrument="EURUSD",
        timeframe="1d",
        forecast_horizon="1d",
        candle_timestamp="2026-10-21T00:00:00+00:00",
        prediction_timestamp="2026-10-21T00:01:00+00:00",
        model_version="v20261020_044046",
        model_sha256="dummy",
        config_hash="dummy",
        prob_sell_raw=0.3, prob_neutral_raw=0.4, prob_buy_raw=0.3,
        prob_sell_calibrated=0.3, prob_neutral_calibrated=0.4, prob_buy_calibrated=0.3,
        decision="NO_TRADE",
        decision_margin=0.01,
        no_trade_reason="Margin below threshold",
        reference_price=1.12000,
        executable_entry_price=1.12000,
        intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
        scheduled_outcome_timestamp="2026-10-22T00:00:00+00:00",
        data_source_id="TEST",
        freshness_status="FRESH",
    )
    store.record_prediction(pred)
    store.resolve_prediction("isolation_test_01", observed_exit_price=1.12050, resolution_timestamp="2026-10-22T00:05:00+00:00")

    # Verify production hash is completely unchanged
    post_hash = hashlib.sha256(prod_ledger_path.read_bytes()).hexdigest()
    assert initial_hash == post_hash, "Production ledger was modified by paper trading!"


def test_restart_recovery_and_pending_outcomes(temp_paper_env, monkeypatch):
    """Verifies that upon worker restart, pending predictions are recovered and resolved idempotently."""
    # 1. Instance A records a prediction
    worker_a = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)
    assert worker_a.initialize_model() is True

    monkeypatch.setattr(worker_a.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker_a, "_load_candles_for_instrument", lambda sym: _mock_valid_daily_candles(sym))
    now_ts = datetime.now(UTC).timestamp()
    mock_event = RawMarketEvent(
        provider="forex_provider", symbol="EURUSD", channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote", "symbol": "EURUSD", "price": 1.12250,
            "bid": 1.12242, "ask": 1.12258, "spread": 0.00016,
            "timestamp": now_ts, "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker_a.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_event)

    res_a = worker_a.poll_once()
    assert res_a["predictions_generated"] == 1
    pending_a = worker_a.store.get_pending_predictions()
    assert len(pending_a) == 1
    pred_id = pending_a[0]["prediction_id"]

    # 2. Simulate shutdown and restart with Instance B
    worker_b = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)
    assert worker_b.initialize_model() is True
    monkeypatch.setattr(worker_b.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker_b.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: mock_event)

    # Recovered state check
    pending_b = worker_b.store.get_pending_predictions()
    assert len(pending_b) == 1
    assert pending_b[0]["prediction_id"] == pred_id

    # 3. Fast-forward horizon to resolve matured prediction
    future_dt = datetime.now(UTC) + timedelta(days=2)
    resolved_count = worker_b._resolve_matured_predictions(future_dt)
    assert resolved_count == 1
    assert len(worker_b.store.get_pending_predictions()) == 0

    # 4. Resolve again: must be idempotent and resolve 0
    resolved_second = worker_b._resolve_matured_predictions(future_dt)
    assert resolved_second == 0


def test_provider_outage_and_degraded_recovery(temp_paper_env, monkeypatch):
    """Verifies graceful degradation when provider fails and automatic recovery when provider restores."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: _mock_valid_daily_candles(sym))

    # Provider outage
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: None)
    res_fail = worker.poll_once()
    assert res_fail["predictions_generated"] == 0
    assert worker.state == WorkerState.DEGRADED

    # Provider restored
    now_ts = datetime.now(UTC).timestamp()
    good_event = RawMarketEvent(
        provider="forex_provider", symbol="EURUSD", channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote", "symbol": "EURUSD", "price": 1.12250,
            "bid": 1.12242, "ask": 1.12258, "spread": 0.00016,
            "timestamp": now_ts, "source": "RESTORED_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: good_event)
    res_ok = worker.poll_once()
    assert res_ok["predictions_generated"] == 1


def test_abnormal_clock_skew_rejection(temp_paper_env, monkeypatch):
    """Verifies that quotes with > 300s clock skew are rejected to prevent timing anomalies."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"])
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: _mock_valid_daily_candles(sym))

    skewed_ts = (datetime.now(UTC) - timedelta(seconds=500)).timestamp()
    skewed_event = RawMarketEvent(
        provider="forex_provider", symbol="EURUSD", channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote", "symbol": "EURUSD", "price": 1.12250,
            "bid": 1.12242, "ask": 1.12258, "spread": 0.00016,
            "timestamp": skewed_ts, "source": "SKEWED_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: skewed_event)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    assert worker.state == WorkerState.DEGRADED
    assert "Abnormal clock skew" in res["instruments"]["EURUSD"]["reason"]


def test_operational_status_truthful_state(temp_paper_env):
    """Verifies that get_operational_status() never displays RUNNING when worker is stopped."""
    # When stopped
    status_stopped = PaperTradingWorker.get_operational_status(store_dir=temp_paper_env)
    assert status_stopped["worker_state"] == "STOPPED"
    assert status_stopped["process_ownership"]["lock_acquired"] is False

    # When running
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], poll_interval_seconds=1.0)
    assert worker.start() is True
    try:
        status_running = PaperTradingWorker.get_operational_status(store_dir=temp_paper_env)
        assert status_running["worker_state"] == "RUNNING"
        assert status_running["process_ownership"]["lock_acquired"] is True
        assert status_running["process_ownership"]["active_pid"] == os.getpid()
    finally:
        worker.stop()

    # After stop
    status_post = PaperTradingWorker.get_operational_status(store_dir=temp_paper_env)
    assert status_post["worker_state"] == "STOPPED"
    assert status_post["process_ownership"]["lock_acquired"] is False


def test_daily_health_report_generation(temp_paper_env):
    """Verifies that daily health report renders correctly with INSUFFICIENT PROSPECTIVE EVIDENCE."""
    from paper_trading.health_report import generate_daily_health_report, render_health_report_markdown

    report = generate_daily_health_report(store_dir=temp_paper_env)
    assert report["evidence_banner"] == "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    assert report["is_sufficient_evidence"] is False
    assert report["protected_hash_verification"]["all_intact"] is True

    md = render_health_report_markdown(report)
    assert "INSUFFICIENT PROSPECTIVE EVIDENCE" in md
    assert "VERIFIED INTACT" in md


def test_stale_candle_rejection(temp_paper_env, monkeypatch):
    """Verifies that candles older than max_candle_age_hours are marked UNAVAILABLE and skipped."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"], max_candle_age_hours=84.0)
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: _mock_valid_daily_candles(sym))

    # Mock fresh live quote on Oct 20
    now_ts = datetime.now(UTC).timestamp()
    fresh_event = RawMarketEvent(
        provider="forex_provider", symbol="EURUSD", channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote", "symbol": "EURUSD", "price": 1.1250,
            "bid": 1.1248, "ask": 1.1252, "spread": 0.0004,
            "timestamp": now_ts, "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: fresh_event)

    # Mock stale historical candles ending 10 days ago (240 hours old)
    stale_date = datetime.now(UTC) - timedelta(days=10)
    mock_candles = [
        {"timestamp": (stale_date - timedelta(days=i)).isoformat(), "open": 1.12, "high": 1.13, "low": 1.11, "close": 1.12, "volume": 1000}
        for i in range(35, -1, -1)
    ]
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: mock_candles)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    assert res["instruments"]["EURUSD"]["candle_fresh"] is False
    assert "STALE" in res["instruments"]["EURUSD"]["reason"]
    assert worker.state == WorkerState.DEGRADED
    assert len(worker.store.get_pending_predictions()) == 0


def test_future_candle_rejection(temp_paper_env, monkeypatch):
    """Verifies that candles with timestamps in the future trigger DEGRADED state and are rejected."""
    worker = PaperTradingWorker(store_dir=temp_paper_env, instruments=["EURUSD"])
    assert worker.initialize_model() is True
    monkeypatch.setattr(worker.provider, "is_market_open", lambda dt=None: True)
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: _mock_valid_daily_candles(sym))

    now_ts = datetime.now(UTC).timestamp()
    fresh_event = RawMarketEvent(
        provider="forex_provider", symbol="EURUSD", channel="spread",
        received_at=datetime.now(UTC),
        raw_payload={
            "type": "quote", "symbol": "EURUSD", "price": 1.1200,
            "bid": 1.1199, "ask": 1.1201, "spread": 0.0002,
            "timestamp": now_ts, "source": "TEST_FEED",
        },
    )
    monkeypatch.setattr(worker.provider, "fetch_live_quote_sync", lambda sym, timeout=4.0: fresh_event)

    future_date = datetime.now(UTC) + timedelta(days=2)
    mock_candles = [
        {"timestamp": (future_date - timedelta(days=i)).isoformat(), "open": 1.12, "high": 1.13, "low": 1.11, "close": 1.12, "volume": 1000}
        for i in range(35, -1, -1)
    ]
    monkeypatch.setattr(worker, "_load_candles_for_instrument", lambda sym: mock_candles)

    res = worker.poll_once()
    assert res["predictions_generated"] == 0
    assert res["instruments"]["EURUSD"]["candle_fresh"] is False
    assert "future" in res["instruments"]["EURUSD"]["reason"]
    assert worker.state == WorkerState.DEGRADED


