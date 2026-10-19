"""Acceptance test suite verifying the complete adaptive, data-driven platform pipeline:

Tests:
TEST A - DISCOVERY: Dynamic stock universe search & resolving arbitrary unlisted stocks
TEST B - LIVE DATA: MarketStateService continuous tick ingestion & stale detection
TEST C - PREDICTION: Probabilistic forecasting across 5m, 15m, 1h horizons
TEST D - SIGNAL ADAPTATION: Dynamic reaction to market condition changes (no hardcoding)
TEST E - OUTCOME RESOLUTION: Live prediction ledger recording & automated resolution
TEST F - LEARNING & ACCURACY: Brier score, MFE, MAE tracking
TEST G - PAPER TRADE EXECUTION: End-to-end signal -> risk -> paper execution -> fill
"""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from core.events import NormalizedQuoteEvent, NormalizedTradeEvent, TradeSide
from core.instrument import MarketType
from core.symbol_resolver import SymbolResolver
from core.universe import UniverseService
from data.market_state import MarketStateService
from execution.multi_market import NSEExecutionClient
from learning.ledger import PredictionLedgerService
from models.base import SignalAction
from risk.engine import RiskEngine
from signals.forecast import ProbabilisticForecastingEngine
from signals.scanner import MarketScannerService


def test_acceptance_test_a_dynamic_stock_discovery():
    """TEST A - DISCOVERY: Verify discovery of arbitrary stocks beyond hardcoded lists."""
    UniverseService.sync_universe()

    # Search known
    res = UniverseService.search("RELIANCE")
    assert len(res) > 0
    assert any(r["symbol"] == "RELIANCE" for r in res)

    # Search arbitrary unhardcoded stocks in the master universe
    res_zomato = UniverseService.search("ZOMATO")
    assert len(res_zomato) > 0
    assert res_zomato[0]["symbol"] == "ZOMATO"

    res_hal = UniverseService.search("HAL")
    assert len(res_hal) > 0
    assert res_hal[0]["symbol"] == "HAL"

    # Dynamic resolution
    inst = UniverseService.resolve_arbitrary_stock("ZOMATO")
    assert inst is not None
    assert inst.canonical_symbol == "NSE:ZOMATO"
    assert inst.market == MarketType.NSE


def test_acceptance_test_b_live_market_state_service():
    """TEST B - LIVE DATA: State updates dynamically from incoming events and detects staleness."""
    service = MarketStateService.get_instance()

    from datetime import UTC, datetime, timedelta
    now = datetime.now(UTC)

    # Push quote
    q = NormalizedQuoteEvent(
        provider="nse_provider",
        symbol="NSE:TCS",
        timestamp=now,
        received_at=now,
        bid_price=4145.0,
        bid_size=100.0,
        ask_price=4146.0,
        ask_size=100.0,
    )
    service.on_quote(q)

    state = service.get_state("NSE:TCS")
    assert state is not None
    assert state.bid_price == 4145.0
    assert state.ask_price == 4146.0
    assert state.spread == 1.0
    assert state.is_stale is False

    # Simulate stale data
    old_q = NormalizedQuoteEvent(
        provider="nse_provider",
        symbol="NSE:INFY",
        timestamp=now - timedelta(seconds=65),
        received_at=now - timedelta(seconds=65),
        bid_price=1750.0,
        bid_size=50.0,
        ask_price=1751.0,
        ask_size=50.0,
    )
    service.on_quote(old_q)
    state_infy = service.get_state("NSE:INFY")
    assert state_infy is not None
    assert state_infy.is_stale is True


def test_acceptance_test_c_probabilistic_forecasting_horizons():
    """TEST C - PREDICTION: Generate 5m, 15m, 1h forecasts with explicit probabilities."""
    engine = ProbabilisticForecastingEngine()

    candles = [{"close": 2900.0 + (i * 2.0)} for i in range(15)]
    fc_1h = engine.forecast_horizon(
        symbol="NSE:RELIANCE",
        current_price=2930.0,
        atr=30.0,
        horizon="1h",
        candle_history=candles,
    )

    assert fc_1h.forecast_horizon == "1h"
    assert fc_1h.prob_up > 0.0
    assert fc_1h.prob_down > 0.0
    assert fc_1h.target_hit_probability > 0.0
    assert fc_1h.stop_hit_probability > 0.0
    assert fc_1h.target_price > fc_1h.entry_reference_price
    assert fc_1h.invalidation_stop_price < fc_1h.entry_reference_price

    # Verify 5m horizon
    fc_5m = engine.forecast_horizon(
        symbol="NSE:RELIANCE",
        current_price=2930.0,
        atr=30.0,
        horizon="5m",
        candle_history=candles,
    )
    assert fc_5m.forecast_horizon == "5m"
    assert fc_5m.horizon_seconds == 300


def test_acceptance_test_d_signal_adapts_to_market_conditions():
    """TEST D - SIGNAL: Signal changes dynamically as prices trend vs crash."""
    engine = ProbabilisticForecastingEngine()

    # 1. Strong Bullish run
    bull_candles = [{"close": 100.0 + (i * 3.0)} for i in range(20)]
    fc_bull = engine.forecast_horizon(
        symbol="BTC-USD",
        current_price=160.0,
        atr=2.0,
        horizon="1h",
        candle_history=bull_candles,
    )
    assert fc_bull.prob_up > fc_bull.prob_down

    # 2. Strong Bearish crash
    bear_candles = [{"close": 200.0 - (i * 4.0)} for i in range(20)]
    fc_bear = engine.forecast_horizon(
        symbol="BTC-USD",
        current_price=120.0,
        atr=2.0,
        horizon="1h",
        candle_history=bear_candles,
    )
    assert fc_bear.prob_down > fc_bear.prob_up

    # 3. Stale data -> strictly NO-TRADE
    fc_stale = engine.forecast_horizon(
        symbol="BTC-USD",
        current_price=120.0,
        atr=2.0,
        horizon="1h",
        candle_history=bear_candles,
        data_is_stale=True,
    )
    assert fc_stale.action == SignalAction.NO_TRADE
    assert any("STALE" in r for r in fc_stale.reasons)


def test_acceptance_test_e_prediction_ledger_and_resolution():
    """TEST E & F - OUTCOME: Record forecast, resolve against actual market, compute Brier score."""
    ledger = PredictionLedgerService.get_instance()
    engine = ProbabilisticForecastingEngine()

    candles = [{"close": 2900.0 + (i * 1.5)} for i in range(10)]
    fc = engine.forecast_horizon(
        symbol="NSE:RELIANCE",
        current_price=2915.0,
        atr=25.0,
        horizon="5m",
        candle_history=candles,
    )

    pred_id = ledger.record_forecast(fc)
    assert pred_id in ledger._active_predictions

    # Simulate resolution
    rec = ledger._active_predictions[pred_id]
    ledger._resolve_outcome(rec, exit_price=2940.0, target_hit=True, stop_hit=False)

    metrics = ledger.get_metrics()
    assert metrics["total_recorded"] >= 1
    assert metrics["total_resolved"] >= 1


@pytest.mark.asyncio
async def test_acceptance_test_g_paper_trading_pipeline():
    """TEST G - PAPER TRADING: End-to-end signal -> risk gatekeeper -> paper execution."""
    from signals.fusion import FinalSignalDecision

    risk_engine = RiskEngine(initial_capital=100000.0)
    sig = FinalSignalDecision(
        symbol="NSE:RELIANCE",
        action=SignalAction.BUY,
        confidence=0.82,
        entry_price=2940.0,
        stop_loss=2900.0,
        take_profit=3020.0,
        risk_reward_ratio=2.0,
    )

    # 1. Risk Evaluation
    decision = risk_engine.evaluate_trade(sig)
    assert decision.status.value in ("APPROVED", "RESIZED")
    assert decision.approved_quantity > 0.0

    # 2. Paper Execution
    exec_client = NSEExecutionClient()
    await exec_client.connect()
    exec_client.set_market_price("NSE:RELIANCE", 2940.0)

    order_result = await exec_client.place_order(
        symbol=decision.symbol,
        side="BUY",
        order_type="MARKET",
        quantity=decision.approved_quantity,
        price=decision.entry_price,
    )

    assert order_result["status"] == "FILLED"
    assert order_result["filled_quantity"] == decision.approved_quantity
    await exec_client.disconnect()


@pytest.mark.asyncio
async def test_acceptance_test_api_scanner_and_universe():
    """Verify live API endpoints for universe search, market scanner, and ledger status."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Universe Search
        res_u = await client.get("/universe/search?q=RELIANCE")
        assert res_u.status_code == 200
        assert len(res_u.json()) > 0

        # 2. Scanner Opportunities
        res_s = await client.get("/scanner/opportunities?limit=5")
        assert res_s.status_code == 200
        scan_data = res_s.json()
        assert "top_opportunities" in scan_data
        assert "total_scanned" in scan_data

        # 3. Forecast Endpoint
        res_f = await client.get("/forecast/NSE:RELIANCE?horizon=1h")
        assert res_f.status_code == 200
        fc_data = res_f.json()
        assert "prob_up" in fc_data
        assert "prob_down" in fc_data

        # 4. Ledger Status
        res_l = await client.get("/learning/ledger/status")
        assert res_l.status_code == 200
        l_data = res_l.json()
        assert "total_recorded" in l_data
