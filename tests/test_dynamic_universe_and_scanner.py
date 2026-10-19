"""End-to-End Verification Suite for Dynamic Multi-Market Universe, Cross-Listing, Quotex Adapter,
Developing-Day Analysis, Time-Aligned Pattern Engine, Segmented Learning, and Market Scanner.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from core.instrument import MarketType
from core.symbol_resolver import SymbolResolver
from core.universe import (
    BSEUniverseManager,
    CanonicalInstrumentRegistry,
    NSEUniverseManager,
    QuotexDiscoveryAdapter,
    UniverseService,
    UniverseSyncService,
)
from data.market_state import MarketStateService
from learning.ledger import PredictionLedgerService
from models.base import SignalAction
from signals.fusion import run_live_intelligence_pipeline
from signals.scanner import MarketScannerService


def test_complete_nse_and_bse_dynamic_universe_and_cross_listing():
    """Verify Sections 1, 2, 3, and 6: Complete NSE & BSE universe, scrip codes, ISIN cross-listing, and sync logs."""
    total = UniverseService.sync_universe()
    assert total >= 8500

    nse_stocks = NSEUniverseManager.get_all_supported_stocks()
    bse_stocks = BSEUniverseManager.get_all_supported_stocks()

    assert len(nse_stocks) >= 3500
    assert len(bse_stocks) >= 5000

    # Verify BSE is NOT limited to 30 stocks in SymbolResolver
    bse_registered = SymbolResolver.get_all_instruments(MarketType.BSE)
    assert len(bse_registered) >= 5000

    # Resolve arbitrary BSE stock by 6-digit scrip code (e.g. ABB = 500002, RELIANCE = 500325)
    bse_abb = SymbolResolver.resolve("500002")
    assert bse_abb is not None
    assert bse_abb.canonical_symbol == "BSE:ABB"
    assert bse_abb.market == MarketType.BSE
    assert bse_abb.metadata.get("scrip_code") == "500002"

    # Verify ISIN-based Canonical Company Identity linking NSE + BSE
    rel_company = CanonicalInstrumentRegistry.get_company("RELIANCE")
    assert rel_company is not None
    assert rel_company.canonical_id.startswith("ISIN:INE")
    assert rel_company.nse_symbol == "NSE:RELIANCE"
    assert rel_company.bse_symbol == "BSE:RELIANCE"
    assert "NSE" in rel_company.exchanges and "BSE" in rel_company.exchanges
    assert CanonicalInstrumentRegistry.cross_listed_count() >= 2500

    # Verify historical symbol change mapping (e.g. MINDTREE -> LTM, ZOMATO -> ETERNAL)
    ltim = SymbolResolver.resolve("MINDTREE")
    assert ltim is not None
    assert ltim.symbol in ("LTM", "LTIM", "MINDTREE")

    # Verify UniverseSyncService audit log
    sync_logs = UniverseSyncService.get_sync_history()
    assert len(sync_logs) >= 1
    assert sync_logs[-1]["records_received"] >= 8500


def test_quotex_discovery_adapter_and_unavailable_otc_gating():
    """Verify Sections 4, 5, and 18: Quotex separates Platform -> Underlying -> Data Source and blocks fake OTC data."""
    qx_records = QuotexDiscoveryAdapter.get_all_records()
    assert len(qx_records) >= 35

    # Authorized FX-backed Quotex instrument
    eurusd_qx = QuotexDiscoveryAdapter.get_record("EURUSD")
    assert eurusd_qx is not None
    assert eurusd_qx.underlying_instrument == "FX:EURUSD"
    assert eurusd_qx.live_data_available is True

    # Proprietary synthetic OTC instrument without authorized public feed
    msft_otc = QuotexDiscoveryAdapter.get_record("MSFT_OTC")
    assert msft_otc is not None
    assert msft_otc.live_data_available is False
    assert msft_otc.trading_availability == "LIVE_DATA_UNAVAILABLE"

    # Running prediction pipeline on QUOTEX:MSFT_OTC must explicitly report 'Quotex live data unavailable'
    res_unavail = run_live_intelligence_pipeline("QUOTEX:MSFT_OTC", timeframe="5m", horizon="5m")
    assert res_unavail["status"] == "QUOTEX_LIVE_DATA_UNAVAILABLE"
    assert res_unavail["decision"].action == SignalAction.NO_TRADE
    assert "Quotex live data unavailable" in res_unavail["decision"].human_reasons


def test_live_data_developing_day_and_prediction_across_nse_bse_quotex():
    """Verify Sections 8, 9, 10, 11, 13, 17, and 22 (items 4-14): Live data, developing day, pattern engine, and ledger."""
    state_svc = MarketStateService.get_instance()

    # 1. NSE stock receives live data & full DevelopingDayState
    nse_res = run_live_intelligence_pipeline("NSE:TCS", timeframe="5m", horizon="1h")
    assert nse_res["status"] == "OK"
    assert nse_res["market_state"].last_price > 0
    dev_day = nse_res["developing_day"]
    for required_key in (
        "open",
        "current_price",
        "high",
        "low",
        "range",
        "gap",
        "volume",
        "relative_volume",
        "vwap",
        "momentum",
        "volatility",
        "trend",
        "support_level",
        "resistance_level",
        "opening_range_high",
        "opening_range_low",
        "breakout_state",
        "pullback_state",
        "consolidation_state",
        "higher_highs_lows",
        "lower_highs_lows",
    ):
        assert required_key in dev_day

    # Verify Pattern Consensus includes P(up), P(down), P(flat), MFE, MAE, hit rates, time to target
    pat = nse_res["pattern_consensus"]
    for pat_key in (
        "prob_up",
        "prob_down",
        "prob_flat",
        "expected_return",
        "mfe_pct",
        "mae_pct",
        "target_hit_rate",
        "stop_hit_rate",
        "time_to_target_bars",
        "time_to_invalidation_bars",
    ):
        assert pat_key in pat

    # Verify Data Quality Gate and Model Strategy
    assert nse_res["data_quality"]["passed"] is True
    assert nse_res["model_strategy"]["hierarchy_tier"] in (
        "INSTRUMENT_SPECIFIC",
        "REGIME_SPECIFIC",
        "SECTOR_SPECIFIC",
        "EXCHANGE_SPECIFIC",
        "GLOBAL_BASELINE",
    )

    # 2. BSE stock receives live data
    bse_state = state_svc.ensure_instrument_state("BSE:RELIANCE", timeframe="5m")
    assert bse_state is not None
    assert bse_state.last_price > 0

    # 3. Quotex supported instrument receives legitimate underlying data
    qx_state = state_svc.ensure_instrument_state("QUOTEX:EURUSD", timeframe="5m")
    assert qx_state is not None
    assert qx_state.last_price > 0

    # 4. Ledger recording, outcome resolution, 13-dimension segmented tracking, and walk-forward challenger
    ledger = PredictionLedgerService.get_instance()
    pred_id = ledger.record_forecast(
        nse_res["forecast"],
        model_id=nse_res["model_strategy"]["selected_model_id"],
        market="NSE",
    )
    resolved = ledger.resolve_prediction_now(pred_id, exit_price=nse_res["market_state"].last_price * 1.003)
    assert resolved is not None
    metrics = ledger.get_metrics()
    seg = metrics["segmented_performance"]
    for dim in (
        "market",
        "exchange",
        "instrument",
        "sector",
        "timeframe",
        "forecast_horizon",
        "market_regime",
        "volatility_regime",
        "volume_regime",
        "confidence_range",
        "model_version",
        "pattern_type",
        "time_of_day",
    ):
        assert dim in seg


@pytest.mark.asyncio
async def test_universe_audit_and_scanner_page_endpoints():
    """Verify Sections 14, 15, 16, 20, and 22 (items 15-17): /universe/audit, /scanner, and /dashboard."""
    scanner = MarketScannerService()
    scan_rep = scanner.scan_market(min_confidence=0.0, horizon="1h", limit=15, include_no_trade=True)
    assert scan_rep.total_universe_size >= 8500
    assert scan_rep.total_scanned >= 10

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r_sum = await ac.get("/universe/summary")
        assert r_sum.status_code == 200
        s_data = r_sum.json()
        assert s_data["total_instruments"] >= 8500
        assert s_data["by_market"]["NSE"] >= 3500
        assert s_data["by_market"]["BSE"] >= 5000

        r_aud = await ac.get("/universe/audit")
        assert r_aud.status_code == 200
        a_data = r_aud.json()
        assert "NSE" in a_data["markets"]
        assert "BSE" in a_data["markets"]
        assert "QUOTEX" in a_data["markets"]
        assert len(a_data["missing_or_unavailable_instruments"]["QUOTEX"]) >= 1

        r_scan_page = await ac.get("/scanner")
        assert r_scan_page.status_code == 200
        assert "Multi-Market Universe Scanner" in r_scan_page.text
        assert "scanExchangeFilter" in r_scan_page.text
        assert "scanSignalFilter" in r_scan_page.text

        r_dash = await ac.get("/dashboard?symbol=BSE:RELIANCE&horizon=15m")
        assert r_dash.status_code == 200
        assert "Market Scanner →" in r_dash.text
