"""Trading Intelligence Platform — Quotex End-to-End Pipeline Verification Test Suite.

Verifies all 9 requirements from Requirement 8:
1. Discover instrument
2. Verify underlying feed connection
3. Fetch live quote
4. Fetch historical candle series
5. Generate features
6. Produce prediction with confidence and trade levels
7. Verify prediction written to ledger
8. Verify desktop application service layer receives prediction
9. Simulate outcome resolution and ledger update
"""

from datetime import UTC, datetime
import pytest

from core.instrument import MarketType
from core.symbol_resolver import SymbolResolver
from core.universe import QuotexInstrumentAdapter, UniverseService
from data.ingestion.quotex import QuotexMarketDataAdapter
from data.market_state import MarketStateService
from desktop.service_controller import DesktopServiceController
from features.pipeline import FeaturePipeline
from learning.ledger import PredictionLedgerService
from models.base import SignalAction
from signals.fusion import run_live_intelligence_pipeline
from core.diagnostics.quotex_audit import run_quotex_pipeline_audit


@pytest.fixture(autouse=True)
def setup_universe():
    if not UniverseService._synced:
        UniverseService.sync_universe()
    else:
        QuotexInstrumentAdapter.discover_instruments()


def test_step_1_quotex_discovery_and_registry():
    """Requirement 8, Step 1: Discover instruments and register in central registry."""
    records = QuotexInstrumentAdapter.get_all_records()
    assert len(records) == 39, f"Expected 39 discovered Quotex instruments, got {len(records)}"

    # Check canonical registration
    eurusd = QuotexInstrumentAdapter.get_record("QUOTEX:EURUSD")
    assert eurusd is not None
    assert eurusd.canonical_symbol == "QUOTEX:EURUSD"
    assert eurusd.underlying_instrument == "FX:EURUSD"
    assert eurusd.live_data_available is True
    assert "5m" in eurusd.supported_horizons

    # Check central registry
    reg_inst = SymbolResolver.resolve("QUOTEX:EURUSD")
    assert reg_inst is not None
    assert reg_inst.market == MarketType.QUOTEX


def test_step_2_underlying_feed_connection_and_otc_gate():
    """Requirement 8, Step 2 & Section 4: Underlying mapping separation and proprietary OTC gating."""
    adapter = QuotexMarketDataAdapter()
    assert adapter.is_live_data_available("EURUSD") is True
    assert adapter.is_live_data_available("BTCUSD_QX") is True
    assert adapter.is_live_data_available("MSFT_OTC") is False

    # Check underlying mapping metadata
    mapping_fx = QuotexInstrumentAdapter.get_underlying_mapping("QUOTEX:EURUSD")
    assert mapping_fx is not None
    assert mapping_fx["underlying_instrument"] == "FX:EURUSD"
    assert "Interbank" in mapping_fx["underlying_provider"]

    mapping_crypto = QuotexInstrumentAdapter.get_underlying_mapping("QUOTEX:BTCUSD_QX")
    assert mapping_crypto is not None
    assert mapping_crypto["underlying_instrument"] == "CRYPTO:BTC-USD"

    # Synthetic OTC must strictly return None quote and empty candles (never fake prices)
    otc_quote = adapter.fetch_live_quote_sync("MSFT_OTC")
    assert otc_quote is None, "Proprietary OTC without public feed must return None for quote"
    otc_candles = adapter.fetch_historical_candles_sync("MSFT_OTC", timeframe="5m")
    assert len(otc_candles) == 0, "Proprietary OTC without public feed must return empty candle list"


def test_step_3_and_4_live_quote_and_historical_candles():
    """Requirement 8, Steps 3 & 4: Fetch verified live quote and multi-timeframe candles."""
    adapter = QuotexMarketDataAdapter()

    # Live quote
    q = adapter.fetch_live_quote_sync("EURUSD")
    assert q is not None, "Failed fetching live quote for QUOTEX:EURUSD"
    assert q.raw_payload["price"] > 0
    assert q.raw_payload["underlying_instrument"] == "FX:EURUSD"
    assert "timestamp" in q.raw_payload

    # Multi-timeframe historical candles
    mtf = adapter.fetch_multi_timeframe_candles_sync("EURUSD", ["5m", "15m", "1h"], limit=25)
    for tf in ("5m", "15m", "1h"):
        assert tf in mtf
        assert len(mtf[tf]) >= 20, f"Expected at least 20 historical bars for {tf}, got {len(mtf[tf])}"
        assert mtf[tf][-1].close > 0


def test_step_5_generate_features():
    """Requirement 8, Step 5: Ingest data into FeaturePipeline and verify feature calculation."""
    state_svc = MarketStateService.get_instance()
    state = state_svc.ensure_instrument_state("QUOTEX:EURUSD", timeframe="5m")
    assert state is not None
    assert state.last_price > 0

    candles = state_svc.get_candles_list(state, "5m")
    assert len(candles) >= 20

    import pandas as pd
    df = pd.DataFrame(candles)
    feat_pipe = FeaturePipeline()
    feat_df = feat_pipe.compute_all_features(df)

    assert "atr_14" in feat_df.columns
    assert "rsi_14" in feat_df.columns
    assert not feat_df.empty


def test_step_6_produce_prediction_with_trade_levels():
    """Requirement 8, Step 6: Produce prediction with confidence and entry/stop/target levels."""
    eval_res = run_live_intelligence_pipeline("QUOTEX:EURUSD", timeframe="5m", horizon="1h")
    assert eval_res["status"] == "OK"
    dec = eval_res["decision"]
    assert dec is not None
    assert dec.action in (SignalAction.BUY, SignalAction.SELL, SignalAction.NO_TRADE)
    assert 0.0 <= dec.confidence <= 1.0
    assert dec.entry_price > 0
    assert dec.stop_loss > 0
    assert dec.take_profit > 0
    assert len(dec.human_reasons) >= 1


def test_step_7_prediction_written_to_ledger():
    """Requirement 8, Step 7: Verify prediction written to PredictionLedgerService."""
    eval_res = run_live_intelligence_pipeline("QUOTEX:EURUSD", timeframe="5m", horizon="1h")
    fc = eval_res["forecast"]
    assert fc is not None

    ledger = PredictionLedgerService.get_instance()
    initial_recorded = ledger.total_recorded
    pred_id = ledger.record_forecast(fc, market="QUOTEX")

    assert pred_id in ledger._active_predictions
    assert ledger.total_recorded == initial_recorded + 1
    active_rec = ledger._active_predictions[pred_id]
    assert active_rec.canonical_symbol == "QUOTEX:EURUSD"
    assert active_rec.entry_price > 0


def test_step_8_desktop_service_layer_receives_prediction():
    """Requirement 8, Step 8: Verify desktop service layer serves Quotex rows and evaluation payload."""
    ctrl = DesktopServiceController()
    rows = ctrl.get_quotex_section_rows(horizon="1h")
    assert len(rows) == 39

    # Find EURUSD row
    eurusd_row = next((r for r in rows if r["symbol"] == "EURUSD"), None)
    assert eurusd_row is not None
    assert eurusd_row["signal"] in ("BUY", "SELL", "NO-TRADE")
    assert eurusd_row["price"] > 0
    assert eurusd_row["live_status"] in (
        "PROXY (FX SPOT UNDERLYING)",
        "PROXY (FX OTC INTERBANK)",
        "PROXY (INTERBANK FX)",
        "VERIFIED UNDERLYING",
    )

    # Find Synthetic OTC row
    msft_row = next((r for r in rows if r["symbol"] == "MSFT_OTC"), None)
    assert msft_row is not None
    assert msft_row["signal"] == "NO-TRADE"
    assert msft_row["live_status"] == "QUOTEX DATA UNAVAILABLE"
    assert "No authorized live OTC feed" in msft_row["rationale"] or "No authorized public market data feed" in msft_row["rationale"]

    # Full beginner evaluation payload
    payload = ctrl.evaluate_instrument("QUOTEX:EURUSD", timeframe="5m", horizon="1h")
    assert payload["canonical_symbol"] == "QUOTEX:EURUSD"
    assert len(payload.get("candles", [])) >= 20
    assert "why_bullets" in payload


def test_step_9_simulate_outcome_resolution_and_ledger_update():
    """Requirement 8, Step 9: Simulate outcome resolution and ledger update."""
    eval_res = run_live_intelligence_pipeline("QUOTEX:EURUSD", timeframe="5m", horizon="1h")
    fc = eval_res["forecast"]
    ledger = PredictionLedgerService.get_instance()

    pred_id = ledger.record_forecast(fc, market="QUOTEX")
    active_rec = ledger._active_predictions[pred_id]
    entry = active_rec.entry_price

    # Simulate exit price moving favorably
    sim_exit = entry * 1.0025 if active_rec.action == SignalAction.BUY else entry * 0.9975
    outcome = ledger.resolve_prediction_now(pred_id, exit_price=sim_exit)

    assert outcome is not None
    assert pred_id not in ledger._active_predictions
    assert outcome["prediction_id"] == pred_id
    assert "brier_score" in outcome
    assert "actual_return_pct" in outcome
    assert "outcome" in outcome


def test_quotex_diagnostic_audit_health():
    """Verify that the diagnostic audit report passes all 10 stages and explicitly marks direct Quotex API as 0 (BLOCKED)."""
    report = run_quotex_pipeline_audit()
    assert report["total_discovered_quotex_instruments"] == 39
    assert report["verified_quotex_platform_feeds_count"] == 0
    assert "BLOCKED" in report["direct_quotex_platform_status"]
    assert report["underlying_market_proxy_feeds_count"] == 34
    assert report["verified_underlying_feed_instruments"] == 34
    assert report["unsupported_otc_instruments_count"] == 5
    assert len(report["stage_health_breakdown"]) == 10
    for st in report["stage_health_breakdown"]:
        assert st["status"] == "WORKING", f"Stage {st['stage']} reported status: {st['status']}"
