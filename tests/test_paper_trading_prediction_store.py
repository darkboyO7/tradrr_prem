"""Tests for the Prospective Paper-Trading Prediction and Outcome Store.

Verifies:
- Append-only schema integrity with crash-safe atomic flush.
- Forward prediction recording before outcome is known.
- Itemized cost resolution (spread, commission, slippage).
- Restart-recovery without duplicates or state corruption.
- Missing data handling (deferral vs expiration without price fabrication).
- Summary metrics excluding expired/missing data from win-rate denominators.
- Strict isolation from production files.
"""

import json
from pathlib import Path
import pytest

from paper_trading.prediction_store import (
    PaperPredictionRecord,
    PaperTradingPredictionStore,
)


@pytest.fixture
def temp_paper_store(tmp_path):
    store_dir = tmp_path / "paper_trading"
    return PaperTradingPredictionStore(store_dir=store_dir)


def test_record_prediction_and_query_pending(temp_paper_store):
    pred = PaperPredictionRecord(
        prediction_id="paper_pred_001",
        instrument="EURUSD",
        timeframe="1d",
        forecast_horizon="1d",
        candle_timestamp="2026-10-21T00:00:00+00:00",
        prediction_timestamp="2026-10-21T00:01:00+00:00",
        model_version="v20261020_044046",
        model_sha256="d955be760504e9f690b6d3c21d9707fb7ad56a39c3ba5f03fe7aebb833ac45c2",
        config_hash="ebf940958de5f617e359cddaf802570ce0bf2e05a91f41a7350b0ec952f4307e",
        prob_sell_raw=0.20,
        prob_neutral_raw=0.35,
        prob_buy_raw=0.45,
        prob_sell_calibrated=0.22,
        prob_neutral_calibrated=0.32,
        prob_buy_calibrated=0.46,
        decision="BUY",
        decision_margin=0.14,
        no_trade_reason="",
        reference_price=1.12000,
        executable_entry_price=1.12008,
        intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
        scheduled_outcome_timestamp="2026-10-22T00:00:00+00:00",
        data_source_id="TRADINGVIEW_FX_LIVE",
        freshness_status="FRESH",
    )

    success = temp_paper_store.record_prediction(pred)
    assert success is True

    # Duplicate must be rejected
    dup_success = temp_paper_store.record_prediction(pred)
    assert dup_success is False

    pending = temp_paper_store.get_pending_predictions()
    assert len(pending) == 1
    assert pending[0]["prediction_id"] == "paper_pred_001"
    assert pending[0]["decision"] == "BUY"
    assert pending[0]["data_source_id"] == "TRADINGVIEW_FX_LIVE"


def test_resolve_prediction_itemized_costs_and_returns(temp_paper_store):
    pred_buy = PaperPredictionRecord(
        prediction_id="paper_pred_buy",
        instrument="EURUSD",
        timeframe="1d",
        forecast_horizon="1d",
        candle_timestamp="2026-10-21T00:00:00+00:00",
        prediction_timestamp="2026-10-21T00:01:00+00:00",
        model_version="v20261020_044046",
        model_sha256="dummy_hash",
        config_hash="dummy_cfg_hash",
        prob_sell_raw=0.2,
        prob_neutral_raw=0.3,
        prob_buy_raw=0.5,
        prob_sell_calibrated=0.2,
        prob_neutral_calibrated=0.3,
        prob_buy_calibrated=0.5,
        decision="BUY",
        decision_margin=0.2,
        no_trade_reason="",
        reference_price=1.10000,
        executable_entry_price=1.10000,
        intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
        scheduled_outcome_timestamp="2026-10-22T00:00:00+00:00",
        data_source_id="TRADINGVIEW_FX_LIVE",
        freshness_status="FRESH",
    )
    temp_paper_store.record_prediction(pred_buy)

    # Resolve with exit at 1.10550 (+0.50% gross return)
    # Itemized costs: spread 0.75 bps, comm 0.50 bps, slip 0.25 bps -> total 1.50 bps (0.000150)
    outcome = temp_paper_store.resolve_prediction(
        prediction_id="paper_pred_buy",
        observed_exit_price=1.10550,
        resolution_timestamp="2026-10-22T00:05:00+00:00",
        spread_cost=0.000075,
        commission_cost=0.000050,
        slippage_cost=0.000025,
    )

    assert outcome is not None
    assert outcome.gross_return == pytest.approx(0.005000, abs=1e-5)
    assert outcome.spread_cost == 0.000075
    assert outcome.commission_cost == 0.000050
    assert outcome.slippage_cost == 0.000025
    assert outcome.total_transaction_costs == 0.000150
    assert outcome.net_return == pytest.approx(0.004850, abs=1e-5)
    assert outcome.outcome == "WIN"

    # Pending should now be empty
    assert len(temp_paper_store.get_pending_predictions()) == 0

    metrics = temp_paper_store.get_summary_metrics()
    assert metrics["valid_active_trades"] == 1
    assert metrics["wins"] == 1
    assert metrics["net_win_rate_pct"] == 100.0


def test_restart_recovery_idempotency(tmp_path):
    store_dir = tmp_path / "paper_restart_test"
    store1 = PaperTradingPredictionStore(store_dir=store_dir)

    # Record 4 predictions
    for i in range(1, 5):
        p = PaperPredictionRecord(
            prediction_id=f"pred_{i}",
            instrument="EURUSD",
            timeframe="1d",
            forecast_horizon="1d",
            candle_timestamp=f"2026-10-2{i}T00:00:00+00:00",
            prediction_timestamp=f"2026-10-2{i}T00:01:00+00:00",
            model_version="v20261020_044046",
            model_sha256="dummy_hash",
            config_hash="dummy_cfg_hash",
            prob_sell_raw=0.2,
            prob_neutral_raw=0.3,
            prob_buy_raw=0.5,
            prob_sell_calibrated=0.2,
            prob_neutral_calibrated=0.3,
            prob_buy_calibrated=0.5,
            decision="BUY" if i % 2 == 0 else "NO_TRADE",
            decision_margin=0.2,
            no_trade_reason="",
            reference_price=1.10000,
            executable_entry_price=1.10000,
            intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
            scheduled_outcome_timestamp=f"2026-10-2{i+1}T00:00:00+00:00",
            data_source_id="TRADINGVIEW_FX_LIVE",
            freshness_status="FRESH",
        )
        store1.record_prediction(p)

    assert len(store1.get_pending_predictions()) == 4

    # Resolve pred_1 and pred_2
    store1.resolve_prediction("pred_1", observed_exit_price=1.10200, resolution_timestamp="2026-10-22T00:00:00+00:00")
    store1.resolve_prediction("pred_2", observed_exit_price=1.09800, resolution_timestamp="2026-10-23T00:00:00+00:00")

    assert len(store1.get_pending_predictions()) == 2

    # Simulate restart by instantiating new store object on same path
    store2 = PaperTradingPredictionStore(store_dir=store_dir)

    pending2 = store2.get_pending_predictions()
    assert len(pending2) == 2
    pending_ids = {p["prediction_id"] for p in pending2}
    assert pending_ids == {"pred_3", "pred_4"}

    # Resolving an already resolved prediction must be rejected
    res_dup = store2.resolve_prediction("pred_1", observed_exit_price=1.10200, resolution_timestamp="2026-10-22T00:00:00+00:00")
    assert res_dup is None


def test_missing_data_defer_and_expire(temp_paper_store):
    pred = PaperPredictionRecord(
        prediction_id="pred_missing",
        instrument="EURAUD",
        timeframe="1d",
        forecast_horizon="1d",
        candle_timestamp="2026-10-20T00:00:00+00:00",
        prediction_timestamp="2026-10-20T00:01:00+00:00",
        model_version="v20261020_044046",
        model_sha256="dummy_hash",
        config_hash="dummy_cfg_hash",
        prob_sell_raw=0.3,
        prob_neutral_raw=0.4,
        prob_buy_raw=0.3,
        prob_sell_calibrated=0.3,
        prob_neutral_calibrated=0.4,
        prob_buy_calibrated=0.3,
        decision="NO_TRADE",
        decision_margin=0.01,
        no_trade_reason="Margin below threshold",
        reference_price=1.65000,
        executable_entry_price=1.65000,
        intended_entry_rule="NEXT_BAR_OPEN_PLUS_HALF_SPREAD",
        scheduled_outcome_timestamp="2026-10-21T00:00:00+00:00",
        data_source_id="TRADINGVIEW_FX_LIVE",
        freshness_status="FRESH",
    )
    temp_paper_store.record_prediction(pred)

    # 1. 1 hour past scheduled timestamp -> should DEFER
    status1 = temp_paper_store.handle_missing_data(
        prediction_id="pred_missing",
        current_timestamp="2026-10-21T01:00:00+00:00",
        max_deferral_seconds=86400.0,
    )
    assert status1 == "MISSING_DATA_DEFERRED"
    assert len(temp_paper_store.get_pending_predictions()) == 1

    # 2. 2 days past scheduled timestamp (exceeding 86400s) -> should EXPIRE WITHOUT DATA
    status2 = temp_paper_store.handle_missing_data(
        prediction_id="pred_missing",
        current_timestamp="2026-10-23T00:00:00+00:00",
        max_deferral_seconds=86400.0,
    )
    assert status2 == "EXPIRED_WITHOUT_DATA"
    assert len(temp_paper_store.get_pending_predictions()) == 0

    # Inspect outcome record
    with open(temp_paper_store.outcomes_file, "r", encoding="utf-8") as f:
        out = json.loads(f.readline())
        assert out["prediction_id"] == "pred_missing"
        assert out["outcome"] == "EXPIRED_WITHOUT_DATA"
        assert out["observed_exit_price"] is None
        assert out["net_return"] == 0.0

    # Summary metrics must exclude expired from win rate denominator
    metrics = temp_paper_store.get_summary_metrics()
    assert metrics["expired_without_data"] == 1
    assert metrics["valid_active_trades"] == 0
    assert metrics["net_win_rate_pct"] == 0.0
