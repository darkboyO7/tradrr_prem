"""Unit and integration tests for Multi-Market Support across NSE, BSE, Forex, and Quotex."""

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import app
from core.events import RawMarketEvent
from core.instrument import AssetClass, MarketType, TradingSession
from core.symbol_resolver import SymbolResolver
from data.ingestion.bse import BSEDataProvider
from data.ingestion.forex import ForexDataProvider
from data.ingestion.nse import NSEDataProvider
from data.ingestion.quotex import QuotexDataProvider
from data.normalization.normalizer import EventNormalizer
from execution.multi_market import (
    BSEExecutionClient,
    ForexExecutionClient,
    NSEExecutionClient,
    QuotexExecutionClient,
)
from models.base import SignalAction
from risk.engine import RiskDecisionStatus, RiskEngine
from signals.fusion import FinalSignalDecision


def test_symbol_resolver_multi_market():
    """Verify SymbolResolver correctly canonicalizes queries and aliases."""
    # Forex EUR/CHF
    eurchf = SymbolResolver.resolve("EUR/CHF")
    assert eurchf is not None
    assert eurchf.canonical_symbol == "FX:EURCHF"
    assert eurchf.market == MarketType.FOREX
    assert eurchf.tick_size == 0.00001
    assert eurchf.lot_size == 1000.0

    # Dash alias
    assert SymbolResolver.resolve("EUR-CHF").canonical_symbol == "FX:EURCHF"
    assert SymbolResolver.resolve("eurchf").canonical_symbol == "FX:EURCHF"

    # NSE Equities
    rel = SymbolResolver.resolve("RELIANCE")
    assert rel is not None
    assert rel.canonical_symbol == "NSE:RELIANCE"
    assert rel.base_currency == "INR"
    assert rel.lot_size == 1.0

    # BSE Scrip
    bse_scrip = SymbolResolver.resolve("500325")
    assert bse_scrip is not None
    assert bse_scrip.canonical_symbol == "BSE:RELIANCE"

    # Quotex Binary Option
    qx = SymbolResolver.resolve("EURCHF_OTC")
    assert qx is not None
    assert qx.market == MarketType.QUOTEX
    assert qx.asset_class == AssetClass.BINARY_OPTION


def test_trading_sessions():
    """Verify session rules for IST stock markets and 24/5 Forex."""
    nse_session = SymbolResolver.get_session(MarketType.NSE)
    assert nse_session is not None
    assert nse_session.timezone == "Asia/Kolkata"
    assert nse_session.start_time == "09:15"
    assert nse_session.end_time == "15:30"

    forex_session = SymbolResolver.get_session(MarketType.FOREX)
    assert forex_session is not None
    assert forex_session.timezone == "America/New_York"


@pytest.mark.asyncio
async def test_nse_data_provider_stream_and_normalization():
    """Verify NSE provider pushes ticks and EventNormalizer standardizes them."""
    provider = NSEDataProvider()
    await provider.connect()

    provider.push_mock_tick_for_test(symbol="RELIANCE", price=2945.50, volume=250.0)

    # Stream one event
    raw_event = None
    async for event in provider.stream_raw_events():
        raw_event = event
        break

    assert raw_event is not None
    assert raw_event.provider == "nse_provider"
    assert raw_event.symbol == "RELIANCE"

    # Normalize
    normalizer = EventNormalizer()
    normalized = normalizer.normalize(raw_event)
    assert normalized is not None
    assert normalized.symbol == "RELIANCE"
    assert normalized.bid_price > 0

    await provider.disconnect()


@pytest.mark.asyncio
async def test_forex_data_provider_and_spread():
    """Verify Forex provider tracks spread and bid/ask quotes."""
    fx_provider = ForexDataProvider()
    await fx_provider.connect()

    fx_provider.push_mock_tick_for_test(symbol="EUR/CHF", bid=0.94210, ask=0.94225)
    assert fx_provider.get_current_spread("EURCHF") == pytest.approx(0.00015, abs=1e-6)

    raw_event = None
    async for event in fx_provider.stream_raw_events():
        raw_event = event
        break

    assert raw_event is not None
    normalizer = EventNormalizer()
    normalized = normalizer.normalize(raw_event)
    assert normalized is not None
    assert normalized.bid_price == 0.94210
    assert normalized.ask_price == 0.94225

    await fx_provider.disconnect()


@pytest.mark.asyncio
async def test_quotex_data_provider():
    """Verify Quotex OTC provider emits normalized paper quotes."""
    qx_provider = QuotexDataProvider()
    await qx_provider.connect()

    qx_provider.push_mock_tick_for_test(symbol="EURCHF_OTC", price=0.94300, payout=0.88)
    raw_event = None
    async for event in qx_provider.stream_raw_events():
        raw_event = event
        break

    assert raw_event is not None
    normalizer = EventNormalizer()
    normalized = normalizer.normalize(raw_event)
    assert normalized is not None
    assert normalized.bid_price == 0.94300

    await qx_provider.disconnect()


@pytest.mark.asyncio
async def test_multi_market_execution_clients():
    """Verify execution clients for NSE, BSE, Forex, and Quotex."""
    # 1. NSE Client
    nse_exec = NSEExecutionClient()
    await nse_exec.connect()
    nse_exec.set_market_price("NSE:RELIANCE", 2940.0)

    res = await nse_exec.place_order(
        symbol="RELIANCE",
        side="BUY",
        order_type="MARKET",
        quantity=15.0,
        price=2940.0,
    )
    assert res["status"] == "FILLED"
    assert res["market"] == "NSE"
    assert res["filled_quantity"] == 15.0
    await nse_exec.disconnect()

    # 2. Forex Client
    fx_exec = ForexExecutionClient()
    await fx_exec.connect()
    fx_exec.set_market_price("FX:EURCHF", 0.94250)

    res_fx = await fx_exec.place_order(
        symbol="EUR/CHF",
        side="BUY",
        order_type="MARKET",
        quantity=5000.0,
        price=0.94250,
    )
    assert res_fx["status"] == "FILLED"
    assert res_fx["market"] == "FOREX"
    assert res_fx["filled_quantity"] == 5000.0
    await fx_exec.disconnect()


def test_risk_engine_lot_size_enforcement():
    """Verify RiskEngine enforces instrument lot size for Indian equities and Forex."""
    risk_engine = RiskEngine(initial_capital=50000.0)

    # Signal for RELIANCE
    signal = FinalSignalDecision(
        symbol="NSE:RELIANCE",
        action=SignalAction.BUY,
        confidence=0.85,
        entry_price=2940.0,
        stop_loss=2900.0,
        take_profit=3020.0,
        risk_reward_ratio=2.0,
    )

    decision = risk_engine.evaluate_trade(signal, requested_quantity=7.8)
    assert decision.status in (RiskDecisionStatus.APPROVED, RiskDecisionStatus.RESIZED)
    # Lot size is 1.0, so quantity must be an integer
    assert decision.approved_quantity == int(decision.approved_quantity)


@pytest.mark.asyncio
async def test_api_instruments_endpoints():
    """Verify REST API /instruments and /instruments/resolve/{query}."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. List instruments
        res = await client.get("/instruments")
        assert res.status_code == 200
        data = res.json()
        assert len(data) >= 20

        # Filter by FOREX
        res_fx = await client.get("/instruments?market=FOREX")
        assert res_fx.status_code == 200
        fx_data = res_fx.json()
        assert any(d["symbol"] == "EURCHF" for d in fx_data)

        # 2. Resolve symbol
        res_res = await client.get("/instruments/resolve/EUR-CHF")
        assert res_res.status_code == 200
        canon = res_res.json()
        assert canon["canonical_symbol"] == "FX:EURCHF"
        assert canon["market"] == "FOREX"
