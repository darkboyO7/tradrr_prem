"""Automated End-to-End Health Tests for the Live Prediction & Outcome Pipeline (Phase 14)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from core.symbol_resolver import SymbolResolver
from core.universe import UniverseService
from execution.multi_market import NSEExecutionClient
from learning.ledger import PredictionLedgerService
from models.base import SignalAction
from risk.engine import RiskDecisionStatus, RiskEngine
from signals.forecast import ProbabilisticForecastingEngine
from signals.fusion import FinalSignalDecision, run_live_intelligence_pipeline


def test_universe_arbitrary_stock_discovery() -> None:
    """Verify full NSE master universe (2,600+ instruments) and arbitrary stock search."""
    total_count = UniverseService.sync_universe()
    assert total_count >= 2000, f"Expected >=2000 instruments, got {total_count}"

    # Test all required arbitrary stocks from Phase 9
    required_queries = ["IRCTC", "DIXON", "TRENT", "POLYCAB", "RVNL", "PERSISTENT", "MRF", "TCS", "RELIANCE"]
    for query in required_queries:
        inst = UniverseService.resolve_arbitrary_stock(query)
        assert inst is not None, f"Failed to resolve arbitrary stock {query}"
        assert inst.canonical_symbol == f"NSE:{query}"
        assert inst.exchange == "NSE"
        assert inst.asset_class.value == "EQUITY"

        # Also verify search returns matches with ISIN and company name
        search_hits = UniverseService.search(query, limit=5)
        assert len(search_hits) >= 1
        assert any(h["canonical_symbol"] == f"NSE:{query}" for h in search_hits)


@pytest.mark.asyncio
async def test_live_prediction_pipeline() -> None:
    """Phase 14 Required Health Test 1: End-to-end live prediction pipeline on NSE:TCS and multi-market."""
    # 1. Run full 14-stage pipeline for NSE:TCS
    res = run_live_intelligence_pipeline("NSE:TCS", timeframe="5m", horizon="15m")
    assert res["status"] == "OK", f"Expected pipeline status OK for NSE:TCS, got {res['status']}"

    # 2. Prove market data is real, non-zero, and NOT the old hardcoded 4150.00 / 45.00 fallback
    ms = res["market_state"]
    assert ms is not None
    assert ms.last_price > 0.0
    assert ms.last_price != 4150.0, "Forbidden static fallback price 4150.00 detected!"
    assert ms.atr_14 > 0.0
    assert ms.atr_14 != 45.0, "Forbidden static fallback ATR 45.00 detected!"
    assert len(ms.rolling_candles_5m) >= 50, f"Expected >=50 candles, got {len(ms.rolling_candles_5m)}"
    if ms.bid_price is not None:
        assert ms.bid_price > 0.0
        assert ms.ask_price is not None and ms.ask_price >= ms.bid_price

    # 3. Prove real technical features are computed from actual OHLCV bars
    features = res["features"]
    for required_feat in [
        "rsi_14",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
        "macd_hist",
        "ema_9",
        "ema_20",
        "ema_50",
        "atr_14",
        "bb_pct_b_20",
        "rvol_20",
    ]:
        assert required_feat in features, f"Missing feature {required_feat}"
        assert features[required_feat] is not None

    # 4. Prove market regime is dynamically classified
    regime = res["regime"]
    assert regime.regime_code not in {"UNKNOWN", "UNCLASSIFIED"}
    assert regime.trend.value in {"STRONG_BULL", "WEAK_BULL", "SIDEWAYS", "WEAK_BEAR", "STRONG_BEAR"}
    assert regime.volatility.value in {"COMPRESSED", "NORMAL", "ELEVATED", "EXTREME"}

    # 5. Prove historical pattern similarity is computed on real bars
    pattern_consensus = res["pattern_consensus"]
    assert pattern_consensus["sample_size"] > 0
    assert 0.0 <= pattern_consensus["win_rate"] <= 1.0
    assert pattern_consensus["direction"] in {"BULLISH", "BEARISH", "NEUTRAL"}

    # 6. Prove ML inference runs and produces calibrated probabilities
    ml_pred = res["ml_prediction"]
    assert ml_pred is not None
    assert 0.0 < ml_pred.probability < 1.0
    raw_scores = ml_pred.raw_scores
    assert abs(raw_scores["buy"] + raw_scores["sell"] + raw_scores["no_trade"] - 1.0) < 1e-2

    # 7. Prove signal fusion probabilities are non-zero and sum to 1.0
    decision = res["decision"]
    assert decision.action in {SignalAction.BUY, SignalAction.SELL, SignalAction.NO_TRADE}
    buy_p = decision.buy_probability or 0.0
    sell_p = decision.sell_probability or 0.0
    nt_p = decision.no_trade_probability or 0.0
    assert buy_p > 0.0 or sell_p > 0.0, "BUY and SELL probabilities cannot both be 0.00!"
    assert abs((buy_p + sell_p + nt_p) - 1.0) < 1e-2

    comp_status = decision.component_status
    for comp_name in ["Technical", "Historical", "ML", "Regime", "Volume"]:
        assert comp_status.get(comp_name) == "AVAILABLE", f"Component {comp_name} not AVAILABLE: {comp_status}"

    # 8. Prove multi-horizon forecasts (5m, 15m, 1h, 4h, 1d)
    fc_engine = ProbabilisticForecastingEngine()
    all_forecasts = fc_engine.forecast_all_horizons(
        symbol="NSE:TCS",
        current_price=ms.last_price,
        atr=ms.atr_14,
        regime=regime,
        technical_signals=res["technical_signals"],
        candle_history=ms.rolling_candles_5m,
        ml_prob_up=raw_scores.get("buy"),
        pattern_consensus=pattern_consensus,
        latest_features=features,
    )
    horizon_map = all_forecasts.forecasts
    for h in ["5m", "15m", "1h", "4h", "1d"]:
        assert h in horizon_map, f"Missing horizon forecast {h}"
        hf = horizon_map[h]
        assert hf.entry_reference_price == pytest.approx(ms.last_price, rel=1e-3)
        assert hf.expected_low < hf.entry_reference_price < hf.expected_high
        assert hf.invalidation_stop_price > 0.0
        assert hf.target_price > 0.0

    # 9. Prove Risk Gatekeeper & Paper Execution integration
    risk_engine = RiskEngine(initial_capital=100000.0)

    # Prove that genuine NO_TRADE is strictly blocked from execution
    if decision.action == SignalAction.NO_TRADE:
        risk_decision_blocked = risk_engine.evaluate_trade(decision, requested_quantity=5.0)
        assert risk_decision_blocked.status == RiskDecisionStatus.REJECTED
        assert "NO_TRADE" in (risk_decision_blocked.rejection_reason or "")

    # Validate execution path on explicit validated trade setup
    test_trade_setup = FinalSignalDecision(
        symbol="NSE:TCS",
        action=SignalAction.BUY,
        confidence=0.75,
        buy_probability=0.75,
        sell_probability=0.15,
        no_trade_probability=0.10,
        entry_price=ms.last_price,
        stop_loss=ms.last_price - (1.5 * ms.atr_14),
        take_profit=ms.last_price + (3.0 * ms.atr_14),
        risk_reward_ratio=2.0,
        regime_code="TRENDING_UP",
        status="APPROVED",
    )
    risk_decision = risk_engine.evaluate_trade(test_trade_setup, requested_quantity=5.0)
    assert risk_decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
    assert risk_decision.approved_quantity >= 1.0

    nse_exec = NSEExecutionClient()
    await nse_exec.connect()
    nse_exec.set_market_price("NSE:TCS", ms.last_price)
    mock_open_session = MagicMock()
    mock_open_session.is_open.return_value = True
    with patch.object(SymbolResolver, "get_session", return_value=mock_open_session):
        order_res = await nse_exec.place_order(
            symbol="NSE:TCS",
            side=test_trade_setup.action.value,
            order_type="MARKET",
            quantity=risk_decision.approved_quantity,
            price=ms.last_price,
        )
    assert order_res["status"] == "FILLED"
    assert order_res["filled_price"] > 0.0
    assert order_res["filled_quantity"] == risk_decision.approved_quantity
    assert order_res["execution_mode"] == "SIMULATED_PAPER"
    await nse_exec.disconnect()

    # 10. Prove generalization across in-scope markets: arbitrary NSE stock, BSE, and QUOTEX
    for multi_sym in ["NSE:IRCTC", "BSE:RELIANCE", "QUOTEX:BTCUSD_QX"]:
        m_res = run_live_intelligence_pipeline(multi_sym, timeframe="5m", horizon="15m")
        assert m_res["status"] == "OK", f"Expected OK for {multi_sym}, got {m_res['status']}"
        assert m_res["market_state"].last_price > 0.0
        m_dec = m_res["decision"]
        assert (m_dec.buy_probability or 0.0) + (m_dec.sell_probability or 0.0) > 0.0

    # Verify that out-of-scope / unconfigured Forex instruments fail closed without fabricating prices
    fx_res = run_live_intelligence_pipeline("FX:EURCHF", timeframe="5m", horizon="15m")
    assert fx_res["status"] == "DATA_UNAVAILABLE"


def test_no_silent_fallbacks_on_invalid_instrument() -> None:
    """Phase 8: Verify invalid instruments return INSTRUMENT_NOT_FOUND instead of fake fallback prices."""
    res = run_live_intelligence_pipeline("NSE:INVALID_NONEXISTENT_TICKER_999", timeframe="5m")
    assert res["status"] == "INSTRUMENT_NOT_FOUND"
    decision = res["decision"]
    assert decision.status == "INSTRUMENT_NOT_FOUND"
    assert decision.buy_probability is None
    assert decision.sell_probability is None


def test_prediction_outcome_pipeline() -> None:
    """Phase 14 Required Health Test 2: Prediction Ledger & Outcome Resolution Pipeline."""
    ledger = PredictionLedgerService()

    # 1. Verify fresh ledger reports None / "N/A (0 Resolved)" — NEVER hardcoded 68.4%
    initial_metrics = ledger.get_metrics()
    assert initial_metrics["total_resolved"] == 0
    assert initial_metrics["prediction_accuracy"] is None
    assert initial_metrics["accuracy_display"] == "N/A (0 Resolved)"

    # 2. Record a live prediction from the real pipeline
    live_res = run_live_intelligence_pipeline("NSE:TCS", timeframe="5m", horizon="15m")
    fc = live_res["forecast"]
    entry_price = fc.entry_reference_price
    assert entry_price > 0.0

    pred_id = ledger.record_forecast(fc, model_id="lgbm_live_v1")
    assert pred_id.startswith("pred_")

    # 3. Resolve prediction with observed market exit price and verify mathematical integrity
    observed_exit_price = entry_price * 1.005
    outcome = ledger.resolve_prediction_now(pred_id, exit_price=observed_exit_price)
    assert outcome is not None
    assert outcome["outcome"] in ("WIN", "LOSS", "NEUTRAL")
    assert "actual_return_pct" in outcome
    assert "brier_score" in outcome
    assert 0.0 <= outcome["brier_score"] <= 1.0
    assert outcome["mfe_pct"] >= 0.0
    assert outcome["mae_pct"] >= 0.0

    # 4. Verify accuracy metrics now reflect the real resolved outcome
    updated_metrics = ledger.get_metrics()
    assert updated_metrics["total_resolved"] == 1
    assert updated_metrics["mean_brier_score"] is not None
    assert updated_metrics["accuracy_display"].endswith("(1 Resolved)")
    assert updated_metrics["expectancy_pct"] is not None


def test_synchronized_developing_day_mtf_and_self_development() -> None:
    """Verify Developing-Day state, Multi-Timeframe (1m-1d) sync, Time-Aligned Similarity, Human Reasons, and Governance."""
    res = run_live_intelligence_pipeline("NSE:TCS", timeframe="5m", horizon="1h")
    assert res["status"] == "OK"

    # 1. Verify Developing-Day Analysis (Market Open -> Current Timestamp T)
    dev_day = res["developing_day"]
    for key in [
        "day_open",
        "previous_close",
        "current_price",
        "current_high",
        "current_low",
        "opening_range_high",
        "opening_range_low",
        "opening_range_breakout",
        "intraday_return_pct",
        "intraday_volatility_pct",
        "session_volume",
        "relative_volume",
        "vwap",
        "price_vs_vwap_pct",
        "support_level",
        "resistance_level",
        "trend_structure",
        "breakout_retest_behavior",
    ]:
        assert key in dev_day, f"Missing developing_day field: {key}"
    assert dev_day["day_open"] > 0.0
    assert dev_day["vwap"] > 0.0
    assert dev_day["current_low"] <= dev_day["current_price"] <= dev_day["current_high"]

    # 2. Verify Multi-Timeframe Synchronization (1m, 5m, 15m, 1h, 4h, 1d)
    mtf = res["mtf_context"]
    assert "timeframes" in mtf
    for tf_name in ["1m", "5m", "15m", "1h", "4h", "1d"]:
        assert tf_name in mtf["timeframes"], f"Missing timeframe {tf_name} in mtf_context"
        assert mtf["timeframes"][tf_name]["bars"] >= 2
        assert mtf["timeframes"][tf_name]["trend"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert mtf["dominant_bias"] in {"BULLISH", "BEARISH", "MIXED"}

    # 3. Verify Time-Aligned Historical Similarity & Zero Future Leakage
    pat = res["pattern_consensus"]
    assert pat["sample_size"] > 0
    assert "time_aligned_matches" in pat
    assert "horizon_summary" in pat
    lat = res["latency_telemetry"]
    assert lat["future_leakage_guard"] == "VERIFIED_TIMESTAMP_LE_T"
    assert lat["total_pipeline_ms"] > 0.0

    # 4. Verify Beginner-Friendly Human-Readable Reasons
    decision = res["decision"]
    assert len(decision.human_reasons) >= 3
    for hr in decision.human_reasons:
        assert "RSI=" not in hr and "ADX=" not in hr

    # 5. Verify Controlled Self-Development, Failure Analysis, Purged Walk-Forward & Governance
    ledger = PredictionLedgerService.get_instance()
    sd_report = ledger.run_self_development_cycle("NSE:TCS")
    assert sd_report["status"] == "COMPLETED"
    assert sd_report["validation_protocol"]["method"] == "Purged Walk-Forward Validation"
    assert sd_report["validation_protocol"]["future_leakage_check"] == "PASSED_ZERO_OVERLAP"
    assert sd_report["validation_protocol"]["embargo_purge_bars"] >= 5
    assert sd_report["governance_verdict"] in {"PROMOTED_CHALLENGER", "RETAINED_CHAMPION"}
    assert "champion_model" in sd_report
    assert "challenger_model" in sd_report
    assert "failure_analysis" in sd_report

